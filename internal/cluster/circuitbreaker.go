package cluster

import (
	"sync"
	"sync/atomic"
	"time"
)

// cbState enumerates circuit breaker states.
type cbState int32

const (
	cbClosed   cbState = 0 // normal operation
	cbOpen     cbState = 1 // failing, reject requests fast
	cbHalfOpen cbState = 2 // probing with one request
)

// CircuitBreaker is a three-state circuit breaker.
// Closed → Open after failureThreshold consecutive failures.
// Open → HalfOpen after resetTimeout elapses.
// HalfOpen → Closed on probe success; → Open on probe failure.
//
// In HalfOpen state exactly ONE goroutine probes; all others
// fail fast immediately (no queueing).
type CircuitBreaker struct {
	mu               sync.Mutex
	state            cbState
	failures         int
	failureThreshold int
	resetTimeout     time.Duration
	openedAt         time.Time
	probing          atomic.Bool // true while a probe is in flight
}

// NewCircuitBreaker creates a CircuitBreaker that opens after threshold
// consecutive failures and attempts reset after resetTimeout.
func NewCircuitBreaker(threshold int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		failureThreshold: threshold,
		resetTimeout:     resetTimeout,
	}
}

// Allow reports whether the caller may proceed.
// Returns (true, done) when allowed; the caller MUST call done(success) after.
// Returns (false, nil) when the breaker is open or a probe is already in flight.
func (cb *CircuitBreaker) Allow() (allowed bool, done func(success bool)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case cbClosed:
		return true, cb.record

	case cbOpen:
		if time.Since(cb.openedAt) < cb.resetTimeout {
			return false, nil
		}
		// Transition to half-open and allow exactly one probe.
		cb.state = cbHalfOpen
		if !cb.probing.CompareAndSwap(false, true) {
			// Another goroutine beat us here — fail fast.
			return false, nil
		}
		return true, cb.record

	case cbHalfOpen:
		// Already half-open; only the current probe may proceed.
		if !cb.probing.CompareAndSwap(false, true) {
			return false, nil
		}
		return true, cb.record
	}
	return false, nil
}

// record is called by the holder of a probe after the operation completes.
func (cb *CircuitBreaker) record(success bool) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if success {
		cb.failures = 0
		cb.state = cbClosed
		cb.probing.Store(false)
		return
	}

	cb.failures++
	cb.probing.Store(false)

	switch cb.state {
	case cbClosed:
		if cb.failures >= cb.failureThreshold {
			cb.trip()
		}
	case cbHalfOpen:
		cb.trip()
	}
}

// trip moves the breaker to Open state.
func (cb *CircuitBreaker) trip() {
	cb.state = cbOpen
	cb.openedAt = time.Now()
}

// State returns the current state as a string (for logging/metrics).
func (cb *CircuitBreaker) State() string {
	cb.mu.Lock()
	s := cb.state
	cb.mu.Unlock()
	switch s {
	case cbClosed:
		return "closed"
	case cbOpen:
		return "open"
	case cbHalfOpen:
		return "half-open"
	}
	return "unknown"
}

// IsOpen returns true when the breaker is open (fast-failing).
func (cb *CircuitBreaker) IsOpen() bool {
	cb.mu.Lock()
	s := cb.state
	cb.mu.Unlock()
	return s == cbOpen
}
