/** Tailwind configuration — the academic design system of SciAgent.

 * The palette is deliberately restrained: a deep ink-blue for primary actions,
 * warm parchment neutrals for the reading surfaces and a single amber accent
 * for the "Agent is thinking" state.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#f5f7fa',
          100: '#e8edf3',
          200: '#cfd9e4',
          300: '#a8b8ca',
          400: '#7890a8',
          500: '#566f8a',
          600: '#405670',
          700: '#33465c',
          800: '#2b3a4c',
          900: '#1d2836',
          950: '#131b26',
        },
        parchment: {
          50: '#fdfcf9',
          100: '#f8f6f0',
          200: '#efece2',
          300: '#e2ddcd',
          400: '#cdc4ae',
          500: '#b3a68c',
        },
        accent: {
          50: '#fff8ed',
          100: '#ffefd4',
          200: '#ffdda8',
          300: '#ffc471',
          400: '#ffa338',
          500: '#f98412',
          600: '#dd6702',
          700: '#b74c06',
          800: '#943b0c',
          900: '#7a310d',
        },
        teal: {
          500: '#2f7f79',
          600: '#266763',
          700: '#1f5250',
        },
      },
      fontFamily: {
        serif: ['Source Serif 4', 'Noto Serif SC', 'Georgia', 'Cambria', 'serif'],
        sans: ['Inter', 'Noto Sans SC', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        'paper': ['1.0625rem', { lineHeight: '1.85' }],
        'paper-lg': ['1.1875rem', { lineHeight: '1.9' }],
      },
      boxShadow: {
        panel: '0 1px 2px rgba(19, 27, 38, 0.04), 0 8px 24px -12px rgba(19, 27, 38, 0.18)',
        'panel-lg': '0 2px 4px rgba(19, 27, 38, 0.05), 0 20px 48px -20px rgba(19, 27, 38, 0.28)',
      },
      keyframes: {
        'fade-in': { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.55' },
        },
        'thinking-dot': {
          '0%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-3px)' },
          '100%': { transform: 'translateY(0)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 220ms ease-out',
        'slide-up': 'slide-up 260ms cubic-bezier(0.22, 1, 0.36, 1)',
        'pulse-soft': 'pulse-soft 1.8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
