import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { Player, PlayerState } from "../audio/engine";

export function usePlayerState(player: Player): PlayerState {
  const subscribe = useCallback((cb: () => void) => player.subscribe(cb), [player]);
  return useSyncExternalStore(subscribe, () => player.state);
}

/** Re-render at most `fps` times per second with the current playback time. */
export function usePlaybackTime(player: Player, fps = 15): number {
  const [t, setT] = useState(player.time);
  const last = useRef(0);
  useEffect(
    () =>
      player.onFrame((time) => {
        const now = performance.now();
        if (now - last.current > 1000 / fps) {
          last.current = now;
          setT(time);
        }
      }),
    [player, fps],
  );
  return t;
}

export function usePoll<T>(fn: () => Promise<T>, ms: number, enabled: boolean, deps: unknown[] = []): T | undefined {
  const [v, setV] = useState<T>();
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const run = async () => {
      try {
        const r = await fn();
        if (alive) setV(r);
      } catch {
        /* retry on next tick */
      }
    };
    void run();
    const h = setInterval(run, ms);
    return () => {
      alive = false;
      clearInterval(h);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ms, ...deps]);
  return v;
}
