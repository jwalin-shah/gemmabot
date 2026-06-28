# Hackathon Rules & FAQ — Gemma 4 24-Hour Hackathon

> Source: Official Cerebras Discord announcement, June 28 2026

## Prizes
| Track | Prize | Channel |
|-------|-------|---------|
| Track 1: Multiverse Agents | $2,000 | #g4hackathon-multiverse-agents |
| Track 2: People's Choice | $2,000 | X/Twitter (tag @Cerebras + @googlegemma) |
| Track 3: Enterprise Impact | $1,000 | #g4hackathon-enterprise-impact |

## Primary Focus
- **Track 1: Multiverse Agents** — best multi-agent + multimodal use case
- **Track 2: People's Choice** — most organic social media impressions

Track 3 is secondary. The core demo should optimize for Track 1 judging and a clear, shareable Track 2 video.

## Timeline
- **Kickoff + Q&A**: Sun June 28, 10:00 AM PT
- **Live support ends**: Sun June 28, 12:30 PM PT
- **Submission deadline**: Mon June 29, 10:00 AM PT
- **Intermittent support overnight**: Limited

## Rate Limits (Official — Not In Docs, Confirmed In Discord)
| Tier | RPM | Token Limit |
|------|-----|-------------|
| Free (public) | 30 RPM | 1M tokens/day |
| **Hackathon elevated** | **100 RPM** | **100K TPM** |
| Pay-as-you-go | 300 RPM | 500K input TPM |
| **Context (elevated)** | **65K MSL** | **32K MCL** |

## Submission Rules
- Submit to multiple tracks (separate Discord post per track)
- Can update/resubmit any time before deadline
- Pre-existing scaffolding allowed. Core project must use Gemma 4 on Cerebras.
- Teams of 2 recommended (any size allowed)

## Demo Video Requirements
- **Max 60 seconds**
- Must show Cerebras speed impact
- **Recommended**: Side-by-side GPU comparison
- No personal/sensitive info visible

## Judging Criteria (Track 1: Multiverse Agents)
| Criterion | Weight | Details |
|-----------|--------|---------|
| Agent Collaboration | High | Effective multi-agent coordination |
| Multimodal Intelligence | High | Meaningful use of Gemma 4 31B with text, images, and video |
| Speed in Action | High | Demonstrates the impact of Cerebras ultra-fast inference |
| Innovation | Medium | Creative, outside-the-box applications using Gemma 4 31B, including physical AI, robotics, embodied agents, or other real-world systems |

Examples called out by the hackathon: Reachy robots, 3D printing workflows, smart manufacturing, autonomous labs, IoT, and other physical-world integrations.

## Judging Criteria (Track 2: People's Choice)
| Criterion | Details |
|-----------|---------|
| Organic Reach | Highest number of organic impressions on Twitter/X; no paid promotion or amplification |
| Community Engagement | Strong likes, comments, reposts, and discussions |
| Content Quality | Clear, compelling, creative showcase of the project |
| Authenticity | Genuine community excitement around the project and Cerebras + Gemma 4 31B |

## Model Details
- **Model ID**: gemma-4-31b (only variant available)
- **Endpoint**: Standard https://api.cerebras.ai — no preview endpoint
- **API Key**: Standard key works with elevated capacity
- **Image format**: Base64 data URIs only (no hosted URLs)
- **Multimodal**: Max 5 images/request, 10MB total
- **Reasoning**: Off by default. Set reasoning_effort to enable.
- **Structured Outputs**: Yes, strict mode available
- **Tool Calling**: Yes, parallel tool calls default on
- **Context**: 65K MSL / 32K MCL elevated
