import type { TutorResponse } from '../api/tutorApi'

interface Props {
  data: TutorResponse
}

function num(value: number | undefined): string {
  return value === undefined ? '—' : String(value)
}

function list(value: string[] | undefined): string {
  return value && value.length ? value.join(', ') : '—'
}

function models(value: Record<string, number> | undefined): string {
  if (!value) return '—'
  const entries = Object.entries(value)
  return entries.length ? entries.map(([m, c]) => `${m}: ${c}`).join(', ') : '—'
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  )
}

export function DiagnosticsPanel({ data }: Props) {
  const d = data.diagnostics || {}
  return (
    <details className="card diagnostics">
      <summary>Diagnostics / debug</summary>
      <div className="diag-body">
        <div className="diag-warnings">
          <strong>warnings</strong>{' '}
          {data.warnings.length === 0 ? (
            <span className="muted">aucun</span>
          ) : (
            <span className="chips">
              {data.warnings.map((w, i) => (
                <span key={i} className="chip warn">{w}</span>
              ))}
            </span>
          )}
        </div>

        <dl className="diag-grid">
          <Row label="retrieval_mode" value={data.retrieval_mode} />
          <Row label="vector_candidates" value={num(d.vector_candidates)} />
          <Row label="keyword_candidates" value={num(d.keyword_candidates)} />
          <Row label="selected_chunk_count" value={num(d.selected_chunk_count)} />
          <Row label="used_chunk_count" value={num(d.used_chunk_count)} />
          <Row label="grounding_terms" value={list(d.grounding_terms)} />
          <Row label="grounding_overlap" value={list(d.grounding_overlap)} />
          <Row label="ready_embedding_models" value={models(d.ready_embedding_models)} />
        </dl>
      </div>
    </details>
  )
}
