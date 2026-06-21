interface Props {
  reason: string | null
}

export function RefusalBanner({ reason }: Props) {
  return (
    <div className="banner refusal" role="alert">
      <strong>
        Le tuteur ne peut pas répondre de manière fiable avec les documents disponibles.
      </strong>
      {reason ? <p className="refusal-reason">Raison : {reason}</p> : null}
    </div>
  )
}
