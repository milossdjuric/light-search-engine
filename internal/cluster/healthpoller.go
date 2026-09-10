package cluster

import (
	"context"
	"log/slog"
	"net/http"
	"time"
)

// HealthPoller polls each shard node's /health endpoint on a fixed interval.
// On a non-2xx response or connection error it records a failure against the
// node's CircuitBreaker so the coordinator stops routing to it proactively,
// before real search traffic reaches the dead shard.
type HealthPoller struct {
	nodes      []*NodeMeta
	getBreaker func(nodeID string) *CircuitBreaker
	interval   time.Duration
	client     *http.Client
	cancel     context.CancelFunc
}

// NewHealthPoller creates a HealthPoller. Call Start to begin polling.
//
//   - nodes      – snapshot of shard NodeMeta (HTTPAddr used for /health call)
//   - getBreaker – returns the CircuitBreaker for a given nodeID (same function
//                  used by cluster.Client so the same breaker is shared)
//   - interval   – how often to poll each node (recommended: 10s in production)
func NewHealthPoller(
	nodes []*NodeMeta,
	getBreaker func(nodeID string) *CircuitBreaker,
	interval time.Duration,
) *HealthPoller {
	return &HealthPoller{
		nodes:      nodes,
		getBreaker: getBreaker,
		interval:   interval,
		client:     &http.Client{Timeout: 5 * time.Second},
	}
}

// Start launches the polling goroutine. Safe to call once.
func (p *HealthPoller) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	p.cancel = cancel
	go p.run(ctx)
}

// Stop stops the polling goroutine. Safe to call after Start.
func (p *HealthPoller) Stop() {
	if p.cancel != nil {
		p.cancel()
	}
}

func (p *HealthPoller) run(ctx context.Context) {
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, node := range p.nodes {
				p.checkNode(ctx, node)
			}
		}
	}
}

func (p *HealthPoller) checkNode(ctx context.Context, node *NodeMeta) {
	cb := p.getBreaker(node.NodeID)
	allowed, done := cb.Allow()
	if !allowed {
		// Already open — don't pile on.
		return
	}
	url := "http://" + node.HTTPAddr + "/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		done(false)
		return
	}
	resp, err := p.client.Do(req)
	if err != nil {
		slog.Debug("healthpoller: shard unreachable", "node", node.NodeID, "err", err)
		done(false)
		return
	}
	resp.Body.Close()
	ok := resp.StatusCode < 500
	if !ok {
		slog.Warn("healthpoller: shard unhealthy", "node", node.NodeID, "status", resp.StatusCode)
	}
	done(ok)
}
