package cluster_test

import (
	"sync"
	"testing"
	"time"

	"search-eval-platform/internal/cluster"
)

// TestCircuitBreakerClosedState verifies a newly created breaker is in the
// closed state and allows requests.
func TestCircuitBreakerClosedState(t *testing.T) {
	cb := cluster.NewCircuitBreaker(3, time.Second)
	allowed, done := cb.Allow()
	if !allowed {
		t.Fatal("new circuit breaker should be in closed state (allowed=true)")
	}
	done(true) // success
	if cb.State() != "closed" {
		t.Errorf("after success: want closed, got %s", cb.State())
	}
}

// TestCircuitBreakerOpensAfterThreshold verifies the breaker transitions to
// open after the configured number of consecutive failures.
func TestCircuitBreakerOpensAfterThreshold(t *testing.T) {
	cb := cluster.NewCircuitBreaker(3, time.Hour)
	for i := 0; i < 3; i++ {
		allowed, done := cb.Allow()
		if !allowed {
			t.Fatalf("iteration %d: should be allowed (closed)", i)
		}
		done(false) // report failure
	}
	if !cb.IsOpen() {
		t.Error("after 3 failures: circuit breaker should be open")
	}
	// Subsequent requests must be rejected.
	allowed, _ := cb.Allow()
	if allowed {
		t.Error("open circuit breaker should not allow requests")
	}
}

// TestCircuitBreakerHalfOpenAfterTimeout verifies the breaker moves to
// half-open after the reset timeout elapses and returns to closed when the
// probe succeeds.
func TestCircuitBreakerHalfOpenAfterTimeout(t *testing.T) {
	cb := cluster.NewCircuitBreaker(1, 50*time.Millisecond)

	// Trip the breaker with a single failure.
	allowed, done := cb.Allow()
	if !allowed {
		t.Fatal("initial allow should succeed")
	}
	done(false)

	if !cb.IsOpen() {
		t.Fatal("breaker should be open after failure")
	}

	// Wait for the reset timeout to elapse.
	time.Sleep(60 * time.Millisecond)

	// First request after timeout is the probe; it must be allowed.
	allowed, done = cb.Allow()
	if !allowed {
		t.Fatal("after reset timeout, probe request should be allowed")
	}
	// Successful probe → breaker returns to closed.
	done(true)
	if cb.State() != "closed" {
		t.Errorf("after successful probe: want closed, got %s", cb.State())
	}
}

// TestCircuitBreakerHalfOpenProbeExclusivity verifies that exactly one
// goroutine is allowed to probe in the half-open state; all others fail fast.
//
// The key to a race-free test: both goroutines capture their "allowed" flag
// and their "done" callback before either calls done. This prevents the first
// goroutine's done(true) from transitioning the breaker back to closed before
// the second goroutine calls Allow(), which would incorrectly grant it access
// as a normal closed-state request rather than as a half-open probe.
func TestCircuitBreakerHalfOpenProbeExclusivity(t *testing.T) {
	cb := cluster.NewCircuitBreaker(1, 10*time.Millisecond)

	// Trip the breaker.
	allowed, done := cb.Allow()
	if !allowed {
		t.Fatal("initial allow should succeed")
	}
	done(false)

	// Allow the reset timeout to expire so the breaker enters half-open.
	time.Sleep(20 * time.Millisecond)

	// Use a barrier so both goroutines call Allow() before either calls done.
	// This ensures we observe the exclusivity during the actual half-open window.
	var (
		mu       sync.Mutex
		allowed1 bool
		allowed2 bool
		done1    func(bool)
		done2    func(bool)
	)

	var readyWg sync.WaitGroup // signals main that both goroutines are blocked
	var releaseWg sync.WaitGroup // main signals goroutines to proceed
	readyWg.Add(2)
	releaseWg.Add(1)

	var resultWg sync.WaitGroup
	resultWg.Add(2)

	go func() {
		defer resultWg.Done()
		a, d := cb.Allow()
		mu.Lock()
		allowed1 = a
		done1 = d
		mu.Unlock()
		readyWg.Done()  // notify main we have our result
		releaseWg.Wait() // wait until main says both are ready
		if d != nil {
			d(true)
		}
	}()
	go func() {
		defer resultWg.Done()
		a, d := cb.Allow()
		mu.Lock()
		allowed2 = a
		done2 = d
		mu.Unlock()
		readyWg.Done()
		releaseWg.Wait()
		if d != nil {
			d(true)
		}
	}()

	// Wait for both goroutines to have called Allow() and stored results.
	readyWg.Wait()
	// Now release both goroutines to call done; but the allowed flags are already
	// captured so the state transition cannot affect our assertions.
	releaseWg.Done()
	resultWg.Wait()

	_ = done1
	_ = done2

	// Exactly one of the two goroutines should have been granted the probe.
	if allowed1 && allowed2 {
		t.Error("both goroutines allowed in half-open state — probe exclusivity violated")
	}
	if !allowed1 && !allowed2 {
		t.Error("neither goroutine was allowed to probe in half-open state")
	}
}

// TestCircuitBreakerProbeFailureReopens verifies that a failed probe causes
// the breaker to reopen immediately.
func TestCircuitBreakerProbeFailureReopens(t *testing.T) {
	cb := cluster.NewCircuitBreaker(1, 10*time.Millisecond)

	// Trip the breaker.
	allowed, done := cb.Allow()
	if !allowed {
		t.Fatal("initial allow should succeed")
	}
	done(false)

	// Wait for half-open window.
	time.Sleep(20 * time.Millisecond)

	// Probe is allowed.
	allowed, done = cb.Allow()
	if !allowed {
		t.Fatal("probe should be allowed after reset timeout")
	}
	// Probe fails → breaker must reopen.
	done(false)
	if !cb.IsOpen() {
		t.Error("after probe failure: circuit breaker should reopen")
	}
}

// TestCircuitBreakerFailureCountResets verifies that a successful request in
// the closed state resets the consecutive failure counter, so a subsequent
// run of failures below the threshold does not trip the breaker.
func TestCircuitBreakerFailureCountResets(t *testing.T) {
	cb := cluster.NewCircuitBreaker(3, time.Hour)

	// Two failures — not yet at threshold.
	for i := 0; i < 2; i++ {
		allowed, done := cb.Allow()
		if !allowed {
			t.Fatalf("iteration %d: should be allowed", i)
		}
		done(false)
	}

	// One success — this should reset the failure counter.
	allowed, done := cb.Allow()
	if !allowed {
		t.Fatal("should still be allowed (below threshold)")
	}
	done(true)

	// Two more failures — must NOT trip (counter was reset to 0 by success).
	for i := 0; i < 2; i++ {
		allowed, done := cb.Allow()
		if !allowed {
			t.Fatalf("post-reset iteration %d: should be allowed", i)
		}
		done(false)
	}

	if cb.IsOpen() {
		t.Error("breaker should still be closed: only 2 consecutive failures after reset")
	}
}
