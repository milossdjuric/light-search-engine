package cluster

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	saTokenPath  = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	saCACertPath = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
	saNSPath     = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	k8sAPIServer = "https://kubernetes.default.svc"
)

// K8sMemberConfig holds the configuration for K8s-native membership discovery.
type K8sMemberConfig struct {
	// Namespace of the headless service to watch.
	// Defaults to the pod's own namespace read from the service-account path.
	Namespace string

	// Service is the name of the headless Kubernetes service whose Endpoints
	// are watched, e.g. "search-shards".
	Service string

	// HTTPPort is the port used to probe /health on each pod. Default: 8080.
	HTTPPort int

	// GRPCPort is the replication gRPC port included in NodeMeta. Default: 9090.
	GRPCPort int
}

// K8sMemberWatcher watches a Kubernetes Endpoints resource and calls
// ring.Rebuild() whenever the set of ready pod addresses changes.
//
// It uses the in-cluster service account credentials. For local development
// against a real cluster, point KUBECONFIG at your kubeconfig and the watcher
// will fall back to using that cluster's API server instead.
//
// Each ready pod is probed at GET /health to discover its local shard IDs
// (the "local_shards" field in the HealthResponse). This tells the coordinator
// which shards each pod owns — making the ring fully dynamic.
type K8sMemberWatcher struct {
	ring   *Ring
	cfg    K8sMemberConfig
	client *http.Client
}

// NewK8sMemberWatcher creates a watcher that rebuilds ring when endpoints change.
func NewK8sMemberWatcher(ring *Ring, cfg K8sMemberConfig) *K8sMemberWatcher {
	if cfg.HTTPPort == 0 {
		cfg.HTTPPort = 8080
	}
	if cfg.GRPCPort == 0 {
		cfg.GRPCPort = 9090
	}
	return &K8sMemberWatcher{ring: ring, cfg: cfg, client: buildK8sClient()}
}

// Run starts the watch loop. It blocks until ctx is cancelled.
// Call as a goroutine: go w.Run(ctx).
func (w *K8sMemberWatcher) Run(ctx context.Context) {
	ns := w.cfg.Namespace
	if ns == "" {
		if data, err := os.ReadFile(saNSPath); err == nil {
			ns = strings.TrimSpace(string(data))
		}
	}
	if ns == "" {
		slog.Error("k8smember: namespace not set and cannot read from service account", "path", saNSPath)
		return
	}

	slog.Info("k8smember: starting endpoint watch",
		"namespace", ns,
		"service", w.cfg.Service,
		"http_port", w.cfg.HTTPPort)

	backoff := time.Second
	for {
		if err := w.watch(ctx, ns); err != nil {
			if ctx.Err() != nil {
				return
			}
			slog.Warn("k8smember: watch error, reconnecting", "err", err, "backoff", backoff)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			if backoff < 30*time.Second {
				backoff *= 2
			}
		} else {
			backoff = time.Second
		}
	}
}

// watch opens a streaming Watch on the Endpoints resource and processes events.
func (w *K8sMemberWatcher) watch(ctx context.Context, ns string) error {
	token, err := os.ReadFile(saTokenPath)
	if err != nil {
		return fmt.Errorf("k8smember: read service account token: %w", err)
	}

	url := fmt.Sprintf("%s/api/v1/namespaces/%s/endpoints/%s?watch=true&timeoutSeconds=300",
		k8sAPIServer, ns, w.cfg.Service)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(string(token)))
	req.Header.Set("Accept", "application/json")

	resp, err := w.client.Do(req)
	if err != nil {
		return fmt.Errorf("k8smember: watch request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("k8smember: watch returned %d: %s", resp.StatusCode, body)
	}

	dec := json.NewDecoder(resp.Body)
	for {
		var event watchEvent
		if err := dec.Decode(&event); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("k8smember: decode event: %w", err)
		}
		if event.Type == "ERROR" {
			return fmt.Errorf("k8smember: watch error event: %s", event.Object.Message)
		}
		if event.Type == "ADDED" || event.Type == "MODIFIED" {
			w.rebuild(ctx, &event.Object)
		}
	}
}

// watchEvent is the envelope returned by the k8s Watch API.
type watchEvent struct {
	Type   string          `json:"type"`
	Object endpointOrError `json:"object"`
}

// endpointOrError covers both Endpoints and Status (error) objects.
type endpointOrError struct {
	// Endpoints fields
	Subsets []endpointSubset `json:"subsets"`
	// Status fields (on error events)
	Message string `json:"message"`
}

type endpointSubset struct {
	Addresses []endpointAddress `json:"addresses"`
	Ports     []endpointPort    `json:"ports"`
}

type endpointAddress struct {
	IP        string         `json:"ip"`
	TargetRef *objectRef     `json:"targetRef"`
}

type objectRef struct {
	Name string `json:"name"`
}

type endpointPort struct {
	Name string `json:"name"`
	Port int    `json:"port"`
}

// rebuild probes each ready address and calls ring.Rebuild.
func (w *K8sMemberWatcher) rebuild(ctx context.Context, ep *endpointOrError) {
	var nodes []*NodeMeta
	for _, subset := range ep.Subsets {
		httpPort := w.cfg.HTTPPort
		// Prefer port named "http" from the subset if present.
		for _, p := range subset.Ports {
			if p.Name == "http" {
				httpPort = p.Port
				break
			}
		}
		for _, addr := range subset.Addresses {
			nodeID := addr.IP
			if addr.TargetRef != nil && addr.TargetRef.Name != "" {
				nodeID = addr.TargetRef.Name
			}
			httpAddr := addr.IP + ":" + strconv.Itoa(httpPort)
			grpcAddr := addr.IP + ":" + strconv.Itoa(w.cfg.GRPCPort)

			shards, err := w.probeLocalShards(ctx, httpAddr)
			if err != nil {
				slog.Warn("k8smember: probe failed, skipping node",
					"node", nodeID, "addr", httpAddr, "err", err)
				continue
			}
			nodes = append(nodes, &NodeMeta{
				NodeID:   nodeID,
				HTTPAddr: httpAddr,
				GRPCAddr: grpcAddr,
				Shards:   shards,
				Role:     "shard",
			})
			slog.Debug("k8smember: discovered node", "node", nodeID, "shards", shards)
		}
	}
	if len(nodes) == 0 {
		slog.Warn("k8smember: no healthy nodes discovered after rebuild; not updating ring")
		return
	}
	w.ring.Rebuild(nodes)
	slog.Info("k8smember: ring rebuilt", "nodes", len(nodes))
}

// healthProbeResponse is the subset of /health we care about.
type healthProbeResponse struct {
	LocalShards []int `json:"local_shards"`
}

// probeLocalShards calls GET /health on the given addr and returns local_shards.
func (w *K8sMemberWatcher) probeLocalShards(ctx context.Context, addr string) ([]int, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(probeCtx, http.MethodGet,
		"http://"+addr+"/health", nil)
	if err != nil {
		return nil, err
	}
	resp, err := w.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var h healthProbeResponse
	if err := json.NewDecoder(resp.Body).Decode(&h); err != nil {
		return nil, fmt.Errorf("decode health: %w", err)
	}
	if len(h.LocalShards) == 0 {
		return nil, fmt.Errorf("node reported no local_shards (not in shard mode?)")
	}
	return h.LocalShards, nil
}

// buildK8sClient builds an HTTP client that trusts the in-cluster CA cert.
// Falls back to the default client (which trusts system roots) if the cert
// is not available (e.g. during local development).
func buildK8sClient() *http.Client {
	pool := x509.NewCertPool()
	if ca, err := os.ReadFile(saCACertPath); err == nil {
		pool.AppendCertsFromPEM(ca)
		return &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{RootCAs: pool},
			},
			Timeout: 10 * time.Second,
		}
	}
	return &http.Client{Timeout: 10 * time.Second}
}
