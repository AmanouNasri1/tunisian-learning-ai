// A quick prompt carries both the question and the filters that match it, so a
// click can't leave contradictory filters selected. Empty string = "Automatique"
// (the field is omitted from the request).
export interface QuickPrompt {
  query: string
  section: string
  subject: string
  chapter: string
}

const PROMPTS: QuickPrompt[] = [
  { query: 'Explique la loi binomiale', section: 'SC_EXP', subject: 'MATH', chapter: 'PROBA' },
  { query: 'Comment étudier une fonction avec la dérivée ?', section: 'MATH', subject: 'MATH', chapter: 'FONCTIONS' },
  { query: 'Explique le circuit RLC', section: 'SC_EXP', subject: 'PHYSIQUE', chapter: 'RLC' },
  { query: 'Explique la génétique récessive', section: 'SC_EXP', subject: 'SVT', chapter: 'GENETIQUE' },
  { query: 'Donne-moi une recette de pizza', section: '', subject: '', chapter: '' },
]

interface Props {
  onPick: (prompt: QuickPrompt) => void
  disabled?: boolean
}

export function QuickPrompts({ onPick, disabled }: Props) {
  return (
    <div className="quick">
      <span className="quick-label">Exemples :</span>
      <div className="chips">
        {PROMPTS.map((prompt) => (
          <button
            key={prompt.query}
            type="button"
            className="chip clickable"
            disabled={disabled}
            onClick={() => onPick(prompt)}
          >
            {prompt.query}
          </button>
        ))}
      </div>
    </div>
  )
}
