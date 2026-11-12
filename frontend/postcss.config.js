/**
 * PostCSS configuration.
 *
 * Vite picks this file up automatically.  Without it Tailwind is never
 * expanded and the production bundle ships the literal `@tailwind`
 * directives, which means the whole application renders unstyled.
 */
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
