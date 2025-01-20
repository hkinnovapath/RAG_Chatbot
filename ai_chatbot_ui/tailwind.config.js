// // module.exports = {
// //   content: [
// //     // "./index.html", "./src/**/*.{js,jsx}"
// //     "./src/**/*.{js,jsx,ts,tsx}",
// //   ],
// //   theme: {
// //     extend: {},
// //   },
// //   plugins: [],
// // };
// module.exports = {
//   content: [
//     "./src/**/*.{html,js,jsx,ts,tsx}", // Update this with your file structure
//   ],
//   theme: {
//     extend: {
//       // You can extend Tailwind's default theme here if needed
//     },
//   },
//   plugins: [
//     // Adding plugin for custom scrollbar if necessary
//     function ({ addComponents }) {
//       addComponents({
//         '.custom-scrollbar::-webkit-scrollbar': {
//           width: '8px', // Makes the scrollbar thinner
//         },
//         '.custom-scrollbar::-webkit-scrollbar-track': {
//           background: 'transparent', // Optional: Make the track transparent
//         },
//         '.custom-scrollbar::-webkit-scrollbar-thumb': {
//           backgroundColor: 'rgba(255, 255, 255, 0.4)', // Light gray for the thumb
//           borderRadius: '4px', // Rounded corners
//           border: '2px solid rgba(255, 255, 255, 0.2)', // Optional: Adds a slight border
//         },
//         '.custom-scrollbar::-webkit-scrollbar-thumb:hover': {
//           backgroundColor: 'rgba(255, 255, 255, 0.6)', // Slightly darker when hovered
//         },
//       });
//     },
//   ],
// }


module.exports = {
  content: [
    "./src/**/*.{html,js,jsx,ts,tsx}", // Update this path as per your project structure
  ],
  theme: {
    extend: {
      // You can add more theme extensions here if needed
    },
  },
  plugins: [
    function ({ addComponents }) {
      addComponents({
        // Custom scrollbar styles
        '.custom-scrollbar::-webkit-scrollbar': {
          width: '8px', // Makes the scrollbar thinner
        },
        '.custom-scrollbar::-webkit-scrollbar-track': {
          background: 'transparent', // Transparent track
        },
        '.custom-scrollbar::-webkit-scrollbar-thumb': {
          backgroundColor: 'transparent', // Transparent thumb by default
          borderRadius: '4px', // Rounded corners
          border: '2px solid rgba(255, 255, 255, 0.2)', // Optional: Adds a slight border
        },
        '.custom-scrollbar::-webkit-scrollbar-thumb:hover': {
          backgroundColor: 'rgba(255, 255, 255, 0.6)', // Slightly darker when hovered
        },
        // Color when scrolling
        '.custom-scrollbar:focus::-webkit-scrollbar-thumb': {
          backgroundColor: 'rgba(255, 255, 255, 0.4)', // Light gray for the thumb when scrolling
        },
      });
    },
  ],
}
