import { useRef, useState } from 'react'
import { askTutor, TutorApiError } from './api/tutorApi'
import type { TutorRequest, TutorResponse } from './api/tutorApi'
import { TutorForm } from './components/TutorForm'
import type { QuickPrompt } from './components/QuickPrompts'
import { AnswerPanel } from './components/AnswerPanel'
import { CitationPanel } from './components/CitationPanel'
import { DiagnosticsPanel } from './components/DiagnosticsPanel'
import { RefusalBanner } from './components/RefusalBanner'

interface SentFilters {
  section: string
  subject: string
  chapter: string
}

function formatFilters(filters: SentFilters): string {
  const parts = [filters.section, filters.subject, filters.chapter].filter(Boolean)
  return parts.length ? parts.join(' · ') : 'Automatique'
}

export default function App() {
  // "" = Automatique (omitted from the request). Default to Auto so a fresh form
  // never sends filters that contradict the question.
  const [query, setQuery] = useState('')
  const [section, setSection] = useState('')
  const [subject, setSubject] = useState('')
  const [chapter, setChapter] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<TutorResponse | null>(null)
  const [usedFilters, setUsedFilters] = useState<SentFilters | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const applyQuickPrompt = (prompt: QuickPrompt) => {
    setQuery(prompt.query)
    setSection(prompt.section)
    setSubject(prompt.subject)
    setChapter(prompt.chapter)
  }

  const submit = async () => {
    const trimmed = query.trim()
    if (!trimmed) {
      setError('Veuillez saisir une question.')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    // Build the payload omitting any "Automatique" (empty) filter entirely.
    const request: TutorRequest = { query: trimmed, provider: 'mock', top_k: 6 }
    if (section) request.section = section
    if (subject) request.subject = subject
    if (chapter) request.chapter = chapter

    setLoading(true)
    setError(null)
    try {
      const data = await askTutor(request, controller.signal)
      setResponse(data)
      setUsedFilters({ section, subject, chapter })
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return // superseded by a newer request or a reset
      }
      setResponse(null)
      setUsedFilters(null)
      setError(err instanceof TutorApiError ? err.message : 'Erreur inattendue.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    abortRef.current?.abort()
    setQuery('')
    setSection('')
    setSubject('')
    setChapter('')
    setResponse(null)
    setUsedFilters(null)
    setError(null)
    setLoading(false)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>BacPilot AI</h1>
          <p className="subtitle">
            Tuteur sourcé pour le Bac tunisien · réponses fondées sur les annales
          </p>
        </div>
        <span className="badge">provider: mock</span>
      </header>

      <main className="layout">
        <section className="col">
          <TutorForm
            query={query}
            section={section}
            subject={subject}
            chapter={chapter}
            loading={loading}
            onQuery={setQuery}
            onSection={setSection}
            onSubject={setSubject}
            onChapter={setChapter}
            onQuickPrompt={applyQuickPrompt}
            onSubmit={submit}
            onReset={reset}
          />
        </section>

        <section className="col">
          {error ? (
            <div className="banner error" role="alert">
              {error}
            </div>
          ) : null}

          {loading ? <div className="card muted">Analyse en cours…</div> : null}

          {response && !loading ? (
            <>
              <div className="filters-used">
                Filtres utilisés : <strong>{formatFilters(usedFilters ?? { section, subject, chapter })}</strong>
              </div>
              {response.refused ? (
                <RefusalBanner reason={response.refusal_reason} />
              ) : (
                <AnswerPanel data={response} />
              )}
              <CitationPanel citations={response.citations} />
              <DiagnosticsPanel data={response} />
            </>
          ) : null}

          {!response && !loading && !error ? (
            <div className="card muted">
              Pose une question pour voir la réponse, les citations et les diagnostics.
            </div>
          ) : null}
        </section>
      </main>

      <footer className="footer">
        MVP tuteur · aucune API payante · démo locale
      </footer>
    </div>
  )
}
