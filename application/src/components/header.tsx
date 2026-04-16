interface HeaderProps {
  onNewGame?: () => void;
  isResetting?: boolean;
}

export default function Header({ onNewGame, isResetting }: HeaderProps) {
  return (
    <header className="bg-stone-900 text-white shadow-md px-6 py-3 sticky top-0 z-50 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="text-3xl">♛</span>
        <h1 className="text-2xl font-black tracking-tighter uppercase hidden sm:block">
          Chess<span className="text-indigo-500">Bot</span>
        </h1>
      </div>
      <button
        type="button"
        onClick={onNewGame}
        disabled={isResetting}
        className="bg-stone-100 hover:bg-white text-stone-900 font-bold py-2 px-4 rounded-lg transition-all uppercase tracking-wider text-xs border border-stone-200 shadow-lg disabled:opacity-50 hover:scale-105 active:scale-95"
      >
        {isResetting ? "Starting..." : "New Game"}
      </button>
    </header>
  );
}
