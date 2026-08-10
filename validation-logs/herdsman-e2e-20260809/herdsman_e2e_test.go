package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	agentmodel "trpc.group/trpc-go/trpc-agent-go/model"
)

type herdsmanE2ECapabilities struct {
	Subfolder       bool `json:"aidaptiv_cache_subfolder"`
	PrefixBoundary  bool `json:"aidaptiv_cache_prefix_boundary"`
	PromptPreflight bool `json:"aidaptiv_cache_prompt_preflight"`
}

type herdsmanE2EProps struct {
	Capabilities herdsmanE2ECapabilities `json:"capabilities"`
}

type herdsmanE2EResult struct {
	Scenario         string                  `json:"scenario"`
	Stream           bool                    `json:"stream"`
	Responses        int                     `json:"responses"`
	PartialResponses int                     `json:"partial_responses"`
	ContentBytes     int                     `json:"content_bytes"`
	ReasoningBytes   int                     `json:"reasoning_bytes"`
	PromptTokens     int                     `json:"prompt_tokens"`
	CachedTokens     int                     `json:"cached_tokens"`
	Capabilities     herdsmanE2ECapabilities `json:"capabilities"`
}

func TestHerdsmanLlamaCppE2E(t *testing.T) {
	baseURL := strings.TrimRight(os.Getenv("HERDSMAN_E2E_BASE_URL"), "/")
	imagePath := os.Getenv("HERDSMAN_E2E_IMAGE")
	if baseURL == "" || imagePath == "" {
		t.Skip("HERDSMAN_E2E_BASE_URL and HERDSMAN_E2E_IMAGE are required")
	}

	caps := readHerdsmanE2ECapabilities(t, baseURL)
	client := New(
		"herdsman-e2e-model",
		WithBaseURL(baseURL+"/v1"),
		WithAIDaptivCacheSubfolder(caps.Subfolder),
		WithAIDaptivCachePrefixBoundary(caps.PrefixBoundary),
		WithAIDaptivCachePromptPreflight(caps.PromptPreflight),
		WithShowToolCallDelta(true),
	)

	scenarios := strings.Split(os.Getenv("HERDSMAN_E2E_SCENARIOS"), ",")
	if len(scenarios) == 1 && strings.TrimSpace(scenarios[0]) == "" {
		scenarios = []string{"text", "single-image", "two-turn-two-image", "document-image"}
	}

	for _, rawScenario := range scenarios {
		scenario := strings.TrimSpace(rawScenario)
		if scenario == "" {
			continue
		}
		if scenario == "two-turn-two-image" {
			firstRequest := herdsmanE2ERequest(t, "first-image-turn", imagePath)
			firstResult := runHerdsmanE2ERequest(t, client, "first-image-turn", firstRequest)
			firstResult.Capabilities = caps
			encoded, err := json.Marshal(firstResult)
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("HERDSMAN_E2E_RESULT %s\n", encoded)
		}
		request := herdsmanE2ERequest(t, scenario, imagePath)
		result := runHerdsmanE2ERequest(t, client, scenario, request)
		result.Capabilities = caps
		encoded, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("HERDSMAN_E2E_RESULT %s\n", encoded)
	}
}

func readHerdsmanE2ECapabilities(t *testing.T, baseURL string) herdsmanE2ECapabilities {
	t.Helper()
	client := &http.Client{Timeout: 10 * time.Second}
	response, err := client.Get(baseURL + "/props")
	if err != nil {
		t.Fatalf("GET /props: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("GET /props status = %d", response.StatusCode)
	}
	var props herdsmanE2EProps
	if err := json.NewDecoder(response.Body).Decode(&props); err != nil {
		t.Fatalf("decode /props: %v", err)
	}
	return props.Capabilities
}

func herdsmanE2ERequest(t *testing.T, scenario, imagePath string) *agentmodel.Request {
	t.Helper()
	maxTokens := 8
	temperature := 0.0
	request := &agentmodel.Request{
		GenerationConfig: agentmodel.GenerationConfig{
			MaxTokens:   &maxTokens,
			Temperature: &temperature,
			Stream:      scenario == "text",
		},
		ExtraFields: map[string]any{
			"cache_prompt":      true,
			"aidaptiv_cache_id": "thread-v1-" + strings.Repeat("7", 64),
			"chat_template_kwargs": map[string]any{
				"enable_thinking": false,
			},
		},
	}

	stableText := strings.Repeat("stable Herdsman conversation context ", 160)
	imagePart := agentmodel.ContentPart{
		Type: agentmodel.ContentTypeImage,
		Image: &agentmodel.Image{
			URL:    imagePath,
			Detail: "auto",
		},
	}

	switch scenario {
	case "text":
		request.Messages = []agentmodel.Message{
			agentmodel.NewSystemMessage("You are a concise local assistant."),
			agentmodel.NewUserMessage(stableText + "Reply with one short sentence."),
		}
	case "single-image":
		question := stableText + "Identify the main visual pattern."
		request.Messages = []agentmodel.Message{
			agentmodel.NewSystemMessage("You inspect images accurately."),
			{
				Role: agentmodel.RoleUser,
				ContentParts: []agentmodel.ContentPart{
					imagePart,
					{Type: agentmodel.ContentTypeText, Text: &question},
				},
			},
		}
	case "first-image-turn":
		firstQuestion := stableText + "Describe the first image."
		request.Messages = []agentmodel.Message{
			agentmodel.NewSystemMessage("You inspect conversation image history accurately."),
			{
				Role: agentmodel.RoleUser,
				ContentParts: []agentmodel.ContentPart{
					{Type: agentmodel.ContentTypeText, Text: &firstQuestion},
					imagePart,
				},
			},
		}
	case "two-turn-two-image":
		firstQuestion := stableText + "Describe the first image."
		secondQuestion := strings.Repeat("comparison context ", 80) + "Compare the second image with the first."
		request.Messages = []agentmodel.Message{
			agentmodel.NewSystemMessage("You inspect conversation image history accurately."),
			{
				Role: agentmodel.RoleUser,
				ContentParts: []agentmodel.ContentPart{
					{Type: agentmodel.ContentTypeText, Text: &firstQuestion},
					imagePart,
				},
			},
			agentmodel.NewAssistantMessage("The first image contains a regular colored pattern."),
			{
				Role: agentmodel.RoleUser,
				ContentParts: []agentmodel.ContentPart{
					{Type: agentmodel.ContentTypeText, Text: &secondQuestion},
					imagePart,
				},
			},
		}
	case "document-image":
		document := []byte("# Herdsman report\n" + strings.Repeat("Persistent document section for cache validation. ", 220))
		question := "Use the attached report and image, then give a short comparison."
		request.Messages = []agentmodel.Message{
			agentmodel.NewSystemMessage("Answer from the attached report and visual evidence."),
			{
				Role: agentmodel.RoleUser,
				ContentParts: []agentmodel.ContentPart{
					{
						Type: agentmodel.ContentTypeFile,
						File: &agentmodel.File{
							Name:     "report.md",
							Data:     document,
							MimeType: "text/markdown",
						},
					},
					imagePart,
					{Type: agentmodel.ContentTypeText, Text: &question},
				},
			},
		}
	default:
		t.Fatalf("unknown scenario %q", scenario)
	}
	return request
}

func runHerdsmanE2ERequest(t *testing.T, client *Model, scenario string, request *agentmodel.Request) herdsmanE2EResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	responses, err := client.GenerateContent(ctx, request)
	if err != nil {
		t.Fatalf("%s GenerateContent: %v", scenario, err)
	}

	result := herdsmanE2EResult{Scenario: scenario, Stream: request.Stream}
	finalChoices := 0
	for response := range responses {
		result.Responses++
		if response == nil {
			continue
		}
		if response.Error != nil {
			t.Fatalf("%s response error: %v", scenario, response.Error)
		}
		if response.IsPartial {
			result.PartialResponses++
		}
		if response.Usage != nil {
			result.PromptTokens = response.Usage.PromptTokens
			result.CachedTokens = response.Usage.PromptTokensDetails.CachedTokens
		}
		if len(response.Choices) > 0 {
			finalChoices += len(response.Choices)
			for _, choice := range response.Choices {
				result.ContentBytes += len(choice.Message.Content)
				result.ReasoningBytes += len(choice.Message.ReasoningContent)
			}
		}
	}
	if result.Responses == 0 || finalChoices == 0 {
		t.Fatalf("%s returned no usable response: %#v", scenario, result)
	}
	return result
}
