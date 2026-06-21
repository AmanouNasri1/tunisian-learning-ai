import type { TutorResponse } from '../api/tutorApi'

interface Props {
  data: TutorResponse
}

export function AnswerPanel({ data }: Props) {
  return (
    <div className="card">
      <div className="meta-row">
        <span className="chip ok">refused = false</span>
        <span className="chip">confiance {data.confidence.toFixed(3)}</span>
        <span className="chip">{data.retrieval_mode}</span>
        <span className="chip">{data.provider}/{data.model_name || '?'}</span>
        {data.interaction_id != null ? (
          <span className="chip">interaction #{data.interaction_id}</span>
        ) : null}
      </div>
      <div className="answer">{data.answer}</div>
    </div>
  )
}
