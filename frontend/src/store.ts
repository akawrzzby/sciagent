/**
 * Global application state.
 *
 * A small Zustand store keeps the cross-page state that several workbenches
 * need: the selected papers, the active paper in the reader and the theme.
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface SelectedPaper {
  id: string
  title: string
}

interface SciState {
  /** Papers ticked in the library (used by graph / review / future work). */
  selected: SelectedPaper[]
  /** The paper currently open in the reader. */
  activePaperId: string | null
  /** The section the reader should scroll to (set by "jump to 3.2.2"). */
  jumpTarget: { sectionId: string; pageNo: number } | null
  theme: 'light' | 'dark'

  togglePaper: (paper: SelectedPaper) => void
  setSelected: (papers: SelectedPaper[]) => void
  clearSelection: () => void
  setActivePaper: (id: string | null) => void
  requestJump: (sectionId: string, pageNo: number) => void
  clearJump: () => void
  toggleTheme: () => void
}

export const useSciStore = create<SciState>()(
  persist(
    (set, get) => ({
      selected: [],
      activePaperId: null,
      jumpTarget: null,
      theme: 'light',

      togglePaper: (paper) =>
        set((s) => ({
          selected: s.selected.some((p) => p.id === paper.id)
            ? s.selected.filter((p) => p.id !== paper.id)
            : [...s.selected, paper],
        })),
      setSelected: (papers) => set({ selected: papers }),
      clearSelection: () => set({ selected: [] }),
      setActivePaper: (id) => set({ activePaperId: id }),
      requestJump: (sectionId, pageNo) => set({ jumpTarget: { sectionId, pageNo } }),
      clearJump: () => set({ jumpTarget: null }),
      toggleTheme: () => {
        const next = get().theme === 'light' ? 'dark' : 'light'
        document.documentElement.classList.toggle('dark', next === 'dark')
        set({ theme: next })
      },
    }),
    {
      name: 'sciagent-state',
      partialize: (s) => ({ selected: s.selected, theme: s.theme }),
      onRehydrateStorage: () => (state) => {
        if (state) document.documentElement.classList.toggle('dark', state.theme === 'dark')
      },
    },
  ),
)
