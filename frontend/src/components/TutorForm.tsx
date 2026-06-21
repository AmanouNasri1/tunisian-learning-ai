import type { FormEvent } from 'react'
import { QuickPrompts } from './QuickPrompts'
import type { QuickPrompt } from './QuickPrompts'

const SECTIONS = ['SC_EXP', 'MATH']
const SUBJECTS = ['MATH', 'PHYSIQUE', 'SVT']
const CHAPTERS = ['PROBA', 'FONCTIONS', 'RLC', 'GENETIQUE']

// "" = Automatique (the field is omitted from the request).
const AUTO_LABEL = 'Automatique'

interface Props {
  query: string
  section: string
  subject: string
  chapter: string
  loading: boolean
  onQuery: (value: string) => void
  onSection: (value: string) => void
  onSubject: (value: string) => void
  onChapter: (value: string) => void
  onQuickPrompt: (prompt: QuickPrompt) => void
  onSubmit: () => void
  onReset: () => void
}

export function TutorForm(props: Props) {
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    props.onSubmit()
  }

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <label className="field">
        <span>Question</span>
        <textarea
          className="textarea"
          rows={4}
          placeholder="Ex : Explique le circuit RLC"
          value={props.query}
          disabled={props.loading}
          onChange={(e) => props.onQuery(e.target.value)}
        />
      </label>

      <QuickPrompts onPick={props.onQuickPrompt} disabled={props.loading} />

      <div className="filters">
        <label className="field">
          <span>Section</span>
          <select value={props.section} disabled={props.loading}
                  onChange={(e) => props.onSection(e.target.value)}>
            <option value="">{AUTO_LABEL}</option>
            {SECTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>

        <label className="field">
          <span>Matière</span>
          <select value={props.subject} disabled={props.loading}
                  onChange={(e) => props.onSubject(e.target.value)}>
            <option value="">{AUTO_LABEL}</option>
            {SUBJECTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>

        <label className="field">
          <span>Chapitre</span>
          <select value={props.chapter} disabled={props.loading}
                  onChange={(e) => props.onChapter(e.target.value)}>
            <option value="">{AUTO_LABEL}</option>
            {CHAPTERS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
      </div>

      <p className="hint">
        Astuce : laisse les filtres sur <strong>Automatique</strong> si tu n'es pas sûr.
      </p>

      <div className="actions">
        <button type="submit" className="btn primary"
                disabled={props.loading || !props.query.trim()}>
          {props.loading ? 'Analyse…' : 'Demander au tuteur'}
        </button>
        <button type="button" className="btn ghost" disabled={props.loading}
                onClick={props.onReset}>
          Réinitialiser
        </button>
      </div>

      <p className="provider-note">
        Fournisseur : <strong>mock</strong> (aucune API payante) · top_k = 6
      </p>
    </form>
  )
}
