// import React, { useEffect, useRef } from "react";

// function ChatBox({ messages }) {
//   const chatBoxRef = useRef(null);

//   // Scroll to the bottom when messages change
//   useEffect(() => {
//     if (chatBoxRef.current) {
//       chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
//     }
//   }, [messages]);

//   return (
//     <div
//       ref={chatBoxRef}
//       className="flex-1 bg-gray-700 p-4 mt-7 m-4 rounded-lg overflow-y-auto custom-scrollbar"
//     >
//       <div className="space-y-6">
//         {messages.length === 0 ? (
//           <div className="flex justify-center items-center text-gray-400 text-4xl py-48">
//             Please Input Your Query.....
//           </div>
//         ) : (
//           messages.map((msg, index) => (
//             <div
//               key={index}
//               className={`flex items-center ${
//                 msg.sender === "User" ? "justify-end" : "justify-start"
//               }`}
//             >
//               {/* Avatar */}
//               {msg.sender !== "User" && (
//                 <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-full flex items-center justify-center mr-3">
//                   🤖
//                 </div>
//               )}

//               {/* Message Bubble */}
//               <div
//                 className={`rounded-md p-3 shadow-lg max-w-[75%] ${
//                   msg.sender === "User"
//                     ? "bg-gradient-to-r bg-gray-600/50 border-2 border-gray-600"
//                     : "bg-gradient-to-r bg-gray-400/50 border-gray-400 "
//                 }`}
//                 style={{
//                   display: "inline-block",
//                   wordWrap: "break-word",
//                 }}
//               >
//                 {msg.text}
//               </div>

//               {/* User Avatar */}
//               {msg.sender === "User" && (
//                 <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-pink-500 text-white rounded-full flex items-center justify-center ml-3">
//                   👤
//                 </div>
//               )}
//             </div>
//           ))
//         )}
//       </div>
//     </div>
//   );
// }

// export default ChatBox;



import React, { useEffect, useRef } from "react";

function ChatBox({ messages, isLoading }) {
  const chatBoxRef = useRef(null);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      ref={chatBoxRef}
      className="flex-1 bg-gray-700 p-4 border border-gray-600 mb-2 m-4 rounded-lg overflow-y-auto custom-scrollbar"
    >
      <div className="space-y-6">
        {messages.length === 0 ? (
          <div className="flex justify-center items-center text-gray-400 text-4xl py-64">
            Please Input Your Query.....
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={index}
              className={`flex items-center ${
                msg.sender === "User" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.sender !== "User" && (
                <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-full flex items-center justify-center mr-3">
                  🤖
                </div>
              )}
              <div
                className={`rounded-md p-3 shadow-lg max-w-[75%] ${
                  msg.sender === "User"
                    ? "bg-gradient-to-r bg-gray-600/50 border-2 border-gray-600"
                    : "bg-gradient-to-r bg-gray-400/50 border-gray-400 "
                }`}
                style={{
                  display: "inline-block",
                  wordWrap: "break-word",
                }}
              >
                {msg.text}
              </div>
              {msg.sender === "User" && (
                <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-pink-500 text-white rounded-full flex items-center justify-center ml-3">
                  👤
                </div>
              )}
            </div>
          ))
        )}

        {/* Show "Generating response..." when waiting for the bot */}
        {isLoading && (
          <div className="flex justify-center items-center text-gray-400 text-xl py-4">
            Generating response...
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatBox;
