/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#14171C",
          800: "#1B1F26",
          700: "#242933",
          600: "#333A47",
        },
        parchment: "#E9E6DE",
        emerald: {
          DEFAULT: "#1F5C4D",
          light: "#2E7D68",
          dim: "#173F35",
        },
        brass: {
          DEFAULT: "#B08D57",
          light: "#C9A876",
        },
      },
      fontFamily: {
        display: ["'Source Serif 4'", "Georgia", "serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
}

