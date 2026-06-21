// Typed client for the existing Django tutor endpoint: POST /api/tutor/ask/.
// Uses a relative URL so the Vite dev proxy forwards to Django (no CORS).

export interface Citation {
  chunk_id: number
  source_object_type: string
  source_object_id: number | null
  section: string | null
  subject: string | null
  year: number | null
  chapter: string | null
}

export interface UsedChunk {
  chunk_id: number
  content_type: string
  score: number
  citation: string
}

export interface Diagnostics {
  vector_candidates?: number
  keyword_candidates?: number
  ready_embedding_models?: Record<string, number>
  selected_chunk_count?: number
  used_chunk_count?: number
  grounding_terms?: string[]
  grounding_overlap?: string[]
  context_warnings?: string[]
  [key: string]: unknown
}

export interface TutorResponse {
  query: string
  answer: string
  answer_language: string
  retrieval_mode: string
  confidence: number
  used_chunks: UsedChunk[]
  citations: Citation[]
  warnings: string[]
  refused: boolean
  refusal_reason: string | null
  provider: string
  model_name: string
  diagnostics: Diagnostics
  interaction_id: number | null
}

export interface TutorRequest {
  query: string
  provider: 'mock'
  section?: string | null
  subject?: string | null
  chapter?: string | null
  top_k?: number
}

export class TutorApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TutorApiError'
  }
}

const BACKEND_HINT =
  'Backend indisponible. Démarrez Django : python manage.py runserver 127.0.0.1:8000'

export async function askTutor(
  request: TutorRequest,
  signal?: AbortSignal,
): Promise<TutorResponse> {
  let response: Response
  try {
    response = await fetch('/api/tutor/ask/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (err) {
    // Re-throw aborts so the caller can ignore them; everything else is "no backend".
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw err
    }
    throw new TutorApiError(BACKEND_HINT)
  }

  const rawText = await response.text()
  let data: unknown = null
  if (rawText) {
    try {
      data = JSON.parse(rawText)
    } catch {
      throw new TutorApiError(
        `Réponse illisible du serveur (JSON invalide, HTTP ${response.status}).`,
      )
    }
  }

  if (!response.ok) {
    const detail =
      data && typeof data === 'object' && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : `Erreur serveur (HTTP ${response.status}).`
    throw new TutorApiError(detail)
  }

  if (!data || typeof data !== 'object' || !('answer' in data) || !('refused' in data)) {
    throw new TutorApiError('Réponse inattendue du serveur (format tuteur manquant).')
  }

  return data as TutorResponse
}
