package adapterutil

import llm "github.com/XiaoConstantine/llm-go"

// FileParts reports where a request carries file parts: in user message
// content, and anywhere else (other roles and tool results).
func FileParts(request llm.Request) (inUserMessages, elsewhere bool) {
	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.Kind == llm.PartFile {
				if message.Role == llm.RoleUser {
					inUserMessages = true
				} else {
					elsewhere = true
				}
			}
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				elsewhere = elsewhere || part.Kind == llm.PartFile
			}
		}
	}
	return inUserMessages, elsewhere
}
