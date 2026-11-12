package catalogmeta

import "testing"

func TestClampRetainedCacheRead(t *testing.T) {
	zero, explicit := 0.0, .025
	for _, test := range []struct {
		name        string
		input, read float64
		upstream    *float64
		want        float64
	}{
		{"lower input", .02, .025, nil, .02},
		{"equal", .025, .025, nil, .025},
		{"higher input", .03, .025, nil, .025},
		{"free input", 0, .025, nil, 0},
		{"explicit invalid", .02, .025, &explicit, .025},
		{"explicit zero", .02, 0, &zero, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ClampRetainedCacheRead(test.input, test.read, test.upstream); got != test.want {
				t.Fatalf("clamped read = %g, want %g", got, test.want)
			}
		})
	}
}
