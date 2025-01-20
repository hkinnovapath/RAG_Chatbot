// import React, { useState, useEffect, useRef } from "react";

// function MessageInput({ onSend }) {
//   const [input, setInput] = useState("");
//   const textareaRef = useRef(null);

//   useEffect(() => {
//     // Auto resize the textarea based on content
//     if (textareaRef.current) {
//       textareaRef.current.style.height = "auto"; // Reset the height to auto to calculate new height
//       textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`; // Set height to scroll height
//     }
//   }, [input]); // Re-run when input changes

//   const handleSend = () => {
//     if (input.trim() !== "") {
//       onSend(input);
//       setInput(""); // Clear input after sending
//     }
//   };

//   const handleKeyPress = (event) => {
//     if (event.key === "Enter" && !event.shiftKey) {
//       event.preventDefault();
//       handleSend();
//     }
//   };

//   return (
//     <div className="flex flex-col bg-gray-800 p-2 mb-3 mx-4 rounded-lg">
//       <textarea
//         ref={textareaRef}
//         value={input}
//         onChange={(e) => setInput(e.target.value)}
//         onKeyDown={handleKeyPress}
//         placeholder="Type your message..."
//         className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg outline-none resize-none custom-scrollbar  "
//         style={{
//           minHeight: "40px",  // Set the minimum height
//           maxHeight: "150px", // Set the maximum height
//           overflowY: "auto",  // Enable vertical scrollbars if content exceeds maxHeight
//         }}
//       />
//       <button
//         onClick={handleSend}
//         className={`mt-1 font-bold text-lg ${
//           input.trim() === ""
//             ? "bg-blue-500 cursor-not-allowed"
//             : "bg-blue-600 "
//         } px-4 py-1 text-white rounded-lg`}
//         disabled={input.trim() === ""}
//       >
//         Send
//       </button>
//     </div>
//   );
// }

// export default MessageInput;


import React, { useState } from "react";

function MessageInput({ onSend, isSending }) {
  const [message, setMessage] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() !== "") {
      onSend(message);
      setMessage(""); // Clear the input after sending
    }
  };

  return (
    <form onSubmit={handleSubmit} className="px-4  pb-3 pt-0">
      <div className="flex items-center">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          disabled={isSending} // Disable input while waiting for response
          placeholder={isSending ? "Please wait ⌚..." : "Type a message 💬..."}
          className="flex-1 p-2 rounded-md bg-gray-800 text-white border border-gray-600"
        />
        <button
          type="submit"
          disabled={isSending} // Disable button while waiting for response
          className="ml-2 px-4 py-2 rounded-md bg-indigo-500 text-white disabled:bg-gray-500"
        >
          Send
        </button>
      </div>
    </form>
  );
}

export default MessageInput;
