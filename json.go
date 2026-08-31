package llm

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"
)

// RepairJSON repairs malformed JSON string literals by escaping raw control
// characters and preserving backslashes that precede invalid JSON escapes.
// Bytes outside string literals are unchanged.
func RepairJSON(value string) string {
	var repaired strings.Builder
	repaired.Grow(len(value))
	inString := false
	for index := 0; index < len(value); index++ {
		character := value[index]
		if !inString {
			repaired.WriteByte(character)
			inString = character == '"'
			continue
		}
		switch character {
		case '"':
			repaired.WriteByte(character)
			inString = false
		case '\\':
			if index+1 >= len(value) {
				repaired.WriteString(`\\`)
				continue
			}
			next := value[index+1]
			if next == 'u' && index+5 < len(value) && fourHexDigits(value[index+2:index+6]) {
				repaired.WriteString(value[index : index+6])
				index += 5
				continue
			}
			if strings.ContainsRune(`"\\/bfnrt`, rune(next)) {
				repaired.WriteByte(character)
				repaired.WriteByte(next)
				index++
				continue
			}
			repaired.WriteString(`\\`)
		default:
			switch character {
			case '\b':
				repaired.WriteString(`\b`)
			case '\f':
				repaired.WriteString(`\f`)
			case '\n':
				repaired.WriteString(`\n`)
			case '\r':
				repaired.WriteString(`\r`)
			case '\t':
				repaired.WriteString(`\t`)
			default:
				if character < 0x20 {
					repaired.WriteString(`\u00`)
					repaired.WriteByte("0123456789abcdef"[character>>4])
					repaired.WriteByte("0123456789abcdef"[character&0x0f])
				} else {
					repaired.WriteByte(character)
				}
			}
		}
	}
	return repaired.String()
}

func fourHexDigits(value string) bool {
	if len(value) != 4 {
		return false
	}
	for _, character := range []byte(value) {
		if character < '0' || character > '9' {
			lower := character | 0x20
			if lower < 'a' || lower > 'f' {
				return false
			}
		}
	}
	return true
}

// ParseJSONWithRepair decodes a complete JSON value into target. It first uses
// strict encoding/json semantics, then retries after RepairJSON only when the
// input changed. target follows json.Unmarshal's pointer requirements.
func ParseJSONWithRepair(value string, target any) error {
	err := json.Unmarshal([]byte(value), target)
	if err == nil {
		return nil
	}
	repaired := RepairJSON(value)
	if repaired == value {
		return err
	}
	return json.Unmarshal([]byte(repaired), target)
}

// ParseStreamingJSON parses a complete or partial JSON value. It returns an
// empty object for empty, null, or unrecoverable input. Partial objects, arrays,
// strings, numbers, booleans, and null values are recovered as far as possible.
func ParseStreamingJSON(value string) any {
	if strings.TrimSpace(value) == "" {
		return map[string]any{}
	}
	var complete any
	if ParseJSONWithRepair(value, &complete) == nil {
		if complete == nil {
			return map[string]any{}
		}
		return complete
	}
	for _, candidate := range []string{value, RepairJSON(value)} {
		parsed, err := newPartialJSONParser(candidate).parse()
		if err == nil {
			if parsed == nil {
				return map[string]any{}
			}
			return parsed
		}
	}
	return map[string]any{}
}

type partialJSONParser struct {
	value  string
	offset int
	depth  int
}

const maxPartialJSONDepth = 1000

func newPartialJSONParser(value string) *partialJSONParser {
	return &partialJSONParser{value: strings.TrimSpace(value)}
}

func (p *partialJSONParser) parse() (any, error) {
	p.skipSpace()
	return p.parseAny()
}

func (p *partialJSONParser) parseAny() (any, error) {
	p.skipSpace()
	if p.offset >= len(p.value) {
		return nil, errors.New("unexpected end of JSON")
	}
	if p.depth >= maxPartialJSONDepth {
		return nil, errors.New("partial JSON nesting exceeds limit")
	}
	p.depth++
	defer func() { p.depth-- }()
	switch p.value[p.offset] {
	case '"':
		return p.parseString()
	case '{':
		return p.parseObject(), nil
	case '[':
		return p.parseArray(), nil
	case 'n':
		return p.parseLiteral("null", nil)
	case 't':
		return p.parseLiteral("true", true)
	case 'f':
		return p.parseLiteral("false", false)
	case 'I':
		return p.parseLiteral("Infinity", math.Inf(1))
	case 'N':
		return p.parseLiteral("NaN", math.NaN())
	case '-':
		if strings.HasPrefix("-Infinity", p.value[p.offset:]) && p.offset+1 < len(p.value) {
			p.offset = min(len(p.value), p.offset+len("-Infinity"))
			return math.Inf(-1), nil
		}
	}
	return p.parseNumber()
}

func (p *partialJSONParser) parseString() (string, error) {
	start := p.offset
	p.offset++
	escaped := false
	for p.offset < len(p.value) {
		character := p.value[p.offset]
		if character == '"' && !escaped {
			p.offset++
			var result string
			if err := json.Unmarshal([]byte(p.value[start:p.offset]), &result); err != nil {
				return "", err
			}
			return result, nil
		}
		if character == '\\' {
			escaped = !escaped
		} else {
			escaped = false
		}
		p.offset++
	}
	end := p.offset
	if escaped {
		end--
	}
	var result string
	if json.Unmarshal([]byte(p.value[start:end]+`"`), &result) == nil {
		return result, nil
	}
	backslash := strings.LastIndexByte(p.value[start:end], '\\')
	if backslash >= 0 && json.Unmarshal([]byte(p.value[start:start+backslash]+`"`), &result) == nil {
		return result, nil
	}
	return "", errors.New("malformed partial JSON string")
}

func (p *partialJSONParser) parseObject() map[string]any {
	p.offset++
	result := make(map[string]any)
	for {
		p.skipSpace()
		if p.offset >= len(p.value) {
			return result
		}
		if p.value[p.offset] == '}' {
			p.offset++
			return result
		}
		key, err := p.parseString()
		if err != nil {
			return result
		}
		p.skipSpace()
		if p.offset >= len(p.value) || p.value[p.offset] != ':' {
			return result
		}
		p.offset++
		parsed, err := p.parseAny()
		if err != nil {
			return result
		}
		result[key] = parsed
		p.skipSpace()
		if p.offset >= len(p.value) {
			return result
		}
		switch p.value[p.offset] {
		case ',':
			p.offset++
		case '}':
			p.offset++
			return result
		default:
			return result
		}
	}
}

func (p *partialJSONParser) parseArray() []any {
	p.offset++
	result := make([]any, 0)
	for {
		p.skipSpace()
		if p.offset >= len(p.value) {
			return result
		}
		if p.value[p.offset] == ']' {
			p.offset++
			return result
		}
		parsed, err := p.parseAny()
		if err != nil {
			return result
		}
		result = append(result, parsed)
		p.skipSpace()
		if p.offset >= len(p.value) {
			return result
		}
		switch p.value[p.offset] {
		case ',':
			p.offset++
		case ']':
			p.offset++
			return result
		default:
			return result
		}
	}
}

func (p *partialJSONParser) parseLiteral(literal string, value any) (any, error) {
	remainder := p.value[p.offset:]
	if !strings.HasPrefix(literal, remainder) && !strings.HasPrefix(remainder, literal) {
		return nil, errors.New("malformed JSON literal")
	}
	p.offset = min(len(p.value), p.offset+len(literal))
	return value, nil
}

func (p *partialJSONParser) parseNumber() (any, error) {
	start := p.offset
	for p.offset < len(p.value) && !strings.ContainsRune(",]} \n\r\t", rune(p.value[p.offset])) {
		p.offset++
	}
	raw := p.value[start:p.offset]
	for raw != "" {
		if number, err := strconv.ParseFloat(raw, 64); err == nil {
			return number, nil
		}
		last := raw[len(raw)-1]
		if last != '.' && last != 'e' && last != 'E' && last != '+' && last != '-' {
			break
		}
		raw = raw[:len(raw)-1]
	}
	return nil, errors.New("malformed JSON number")
}

func (p *partialJSONParser) skipSpace() {
	for p.offset < len(p.value) && strings.ContainsRune(" \n\r\t", rune(p.value[p.offset])) {
		p.offset++
	}
}
