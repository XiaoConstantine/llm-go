package catalogmeta

import "errors"

// ValidateCacheReadRate rejects cache-read prices above the uncached input rate.
// Apply it after merging rates; it does not change explicit upstream prices.
func ValidateCacheReadRate(input, cacheRead float64) error {
	if cacheRead > input {
		return errors.New("cache_read must not exceed input")
	}
	return nil
}

// ClampRetainedCacheRead caps an inherited cache-read rate at the merged input
// rate when upstream omits cache_read. Explicit upstream rates are left intact
// for validation by the caller. Use after merging base or tier rates.
func ClampRetainedCacheRead(input, cacheRead float64, upstream *float64) float64 {
	if upstream == nil && cacheRead > input {
		return input
	}
	return cacheRead
}
