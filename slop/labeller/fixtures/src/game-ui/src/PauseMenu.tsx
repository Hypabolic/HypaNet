import { useGame } from "../game";
import styles from "./PauseMenu.module.css";

type Props = { onResume: () => void; onOpen: (panel: string) => void; onRestart: () => void };

export function PauseMenu({ onResume, onOpen, onRestart }: Props) {
  const { paused } = useGame();
  if (!paused) return null;
  return (
    <div className={styles.overlay}>
      <header className="flex flex-col items-center gap-2">
        <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">OPERATION ON HOLD</p>
        <h1 className="text-5xl font-bold uppercase">PAUSED</h1>
        <p className="text-sm text-muted-foreground">Resume recaptures the mouse.</p>
      </header>
      <nav className="mt-8 flex flex-col gap-3">
        <button onClick={onResume}>RESUME</button>
        <button onClick={() => onOpen("settings")}>SETTINGS</button>
        <button onClick={() => onOpen("controls")}>CONTROLS</button>
        <button onClick={onRestart}>RESTART MISSION</button>
      </nav>
    </div>
  );
}
