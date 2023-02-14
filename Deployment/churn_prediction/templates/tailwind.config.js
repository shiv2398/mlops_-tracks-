/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/style.css*/*.{html}"],
  theme: {
    
    screens :{

    sm:'480px',
    md:'768px',
    lg:'976px',
    xl:'1440px'
    },

    extend: {
      colors: {
        transparent: 'transparent',
        current: 'currentColor',
        'white': '#ffffff',
        'purple': '#3f3cbb',
        'midnight': '#121063',
        'metal': '#565584',
        'tahiti': '#3ab7bf',
        'silver': '#ecebff',
        'bubble-gum': '#ff77e9',
        'bermuda': '#78dcca',
      },
    },
  },
  plugins: [],
}