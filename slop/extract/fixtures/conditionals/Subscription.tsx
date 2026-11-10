export function Subscription({ active, trial, plan, nickname }: Props) {
  return (
    <div>
      <h2>{active ? "Your membership" : "Join the co-op"}</h2>
      {trial && <p className="text-sm">Your trial ends on Friday.</p>}
      {plan ? <p>You are on the {plan.name} plan.</p> : <p>Pick a plan to get started.</p>}
      <span>{nickname ?? "Anonymous gardener"}</span>
      <button>{active ? "Manage" : "Join"}</button>
    </div>
  );
}
