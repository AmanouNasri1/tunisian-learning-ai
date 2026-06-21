import type { Citation } from '../api/tutorApi'

interface Props {
  citations: Citation[]
}

function formatCitation(c: Citation): string {
  return [
    `chunk#${c.chunk_id}`,
    c.source_object_type || '—',
    c.year != null ? String(c.year) : '—',
    c.section || '—',
    c.subject || '—',
    c.chapter || '—',
  ].join(' · ')
}

export function CitationPanel({ citations }: Props) {
  return (
    <div className="card">
      <h3 className="card-title">Citations ({citations.length})</h3>
      {citations.length === 0 ? (
        <p className="muted">Aucune citation utilisée.</p>
      ) : (
        <ul className="citations">
          {citations.map((c) => (
            <li key={c.chunk_id} className="citation">
              {formatCitation(c)}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
