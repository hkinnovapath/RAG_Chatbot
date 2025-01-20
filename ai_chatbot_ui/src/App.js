// // import React, { useState } from "react";
// // import axios from "axios";
// // import Sidebar from "./components/Sidebar";
// // import ChatBox from "./components/ChatBox";
// // import MessageInput from "./components/MessageInput";

// // function App() {
// //   const [isSidebarVisible, setIsSidebarVisible] = useState(true);
// //   const [chats, setChats] = useState([]); // Stores chat titles dynamically
// //   const [activeChat, setActiveChat] = useState(null); // Track the current active chat
// //   const [messages, setMessages] = useState([]); // Messages for the active chat
// //   const [firstMessageSent, setFirstMessageSent] = useState(false); // Track if the first message has been sent
// //   const [isLoading, setIsLoading] = useState(false); // Track loading state

// //   const handleSend = async (message) => {
// //     const updatedMessages = [...messages, { sender: "User", text: message }];
// //     setMessages(updatedMessages);

// //     setIsLoading(true); // Show "Generating response..." message

// //     try {
// //       const response = await axios.post("http://localhost:8000/chat", null, {
// //         params: { query: message },
// //       });

// //       console.log("Response from backend:", response.data);

// //       if (response.data && response.data.response) {
// //         const botReply = response.data.response;
// //         setMessages((prevMessages) => [
// //           ...prevMessages,
// //           { sender: "ChatBot", text: botReply },
// //         ]);
// //       } else {
// //         throw new Error("Invalid response format");
// //       }
// //     } catch (error) {
// //       console.error("Error calling the backend:", error);
// //       setMessages((prevMessages) => [
// //         ...prevMessages,
// //         { sender: "ChatBot", text: "Something went wrong. Please try again." },
// //       ]);
// //     } finally {
// //       setIsLoading(false); // Hide the "Generating response..." message
// //     }
// //   };

// //   const toggleSidebar = () => {
// //     setIsSidebarVisible(!isSidebarVisible); // Toggle sidebar visibility
// //   };

// //   const addNewChat = () => {
// //     const newChatTitle = `Chat ${chats.length + 1}`;
// //     setChats([...chats, newChatTitle]);
// //     setActiveChat(newChatTitle); // Automatically switch to the new chat
// //     setMessages([]); // Clear any existing messages for the new chat
// //     setFirstMessageSent(false); // Reset the first message flag
// //   };

// //   const switchChat = (chat) => {
// //     setActiveChat(chat);
// //     setMessages([]); // Reset messages for simplicity
// //     setFirstMessageSent(false); // Reset the first message flag
// //   };

// //   const deleteChat = (chatToDelete) => {
// //     const updatedChats = chats.filter((chat) => chat !== chatToDelete);
// //     setChats(updatedChats);

// //     // If the deleted chat is active, reset the active chat
// //     if (chatToDelete === activeChat) {
// //       setActiveChat(null);
// //       setMessages([]);
// //       setFirstMessageSent(false);
// //     }
// //   };

// //   return (
// //     <div className="flex h-screen bg-gray-900 text-white">
// //       {/* Sidebar */}
// //       <div
// //         className={`transition-all duration-500 ease-in-out fixed top-0 left-0 h-full bg-gray-800 w-64 shadow-lg z-10 transform ${
// //           isSidebarVisible ? "translate-x-0" : "-translate-x-[90%]"
// //         }`}
// //         onClick={() => {
// //           if (!isSidebarVisible) {
// //             setIsSidebarVisible(true); // Expand if the sidebar is currently hidden
// //           }
// //         }} // Sidebar expands if clicked when hidden
// //       >
// //         <Sidebar
// //           chats={chats}
// //           activeChat={activeChat}
// //           onAddNewChat={addNewChat}
// //           onSwitchChat={switchChat}
// //           onDeleteChat={deleteChat}
// //           onToggleSidebar={toggleSidebar} // Passing toggleSidebar function
// //         />
// //       </div>
// //       {/* Chat Area */}
// //       <div
// //         className={`flex-1 transition-all duration-500 ${
// //           isSidebarVisible ? "ml-64" : "ml-10"
// //         }`} // Remove margin when sidebar is hidden
// //       >
// //         <div className="flex flex-col h-full">
// //           {/* ChatBox Display */}
// //           <ChatBox messages={messages} isLoading={isLoading} />
// //           {/* Message Input */}
// //           <MessageInput onSend={handleSend} />
// //         </div>
// //       </div>
// //     </div>
// //   );
// // }

// // export default App;




// import React, { useState, useEffect } from "react";
// import axios from "axios";
// import Sidebar from "./components/Sidebar";
// import ChatBox from "./components/ChatBox";
// import MessageInput from "./components/MessageInput";

// function App() {
//   const [isSidebarVisible, setIsSidebarVisible] = useState(true);
//   const [chats, setChats] = useState([]); // Stores chat titles dynamically
//   const [activeChat, setActiveChat] = useState(null); // Track the current active chat
//   const [messages, setMessages] = useState([]); // Messages for the active chat
//   const [firstMessageSent, setFirstMessageSent] = useState(false); // Track if the first message has been sent
//   const [isLoading, setIsLoading] = useState(false); // Track loading state

//   // Load data from session storage
//   useEffect(() => {
//     const savedChats = JSON.parse(sessionStorage.getItem("chats"));
//     const savedMessages = JSON.parse(sessionStorage.getItem("messages"));
//     if (savedChats) {
//       setChats(savedChats);
//     }
//     if (savedMessages) {
//       setMessages(savedMessages);
//     }
//   }, []);

//   // Save data to session storage when chats or messages change
//   useEffect(() => {
//     sessionStorage.setItem("chats", JSON.stringify(chats));
//     sessionStorage.setItem("messages", JSON.stringify(messages));
//   }, [chats, messages]);

//   const handleSend = async (message) => {
//     const updatedMessages = [...messages, { sender: "User", text: message }];
//     setMessages(updatedMessages);

//     setIsLoading(true); // Show "Generating response..." message

//     try {
//       const response = await axios.post("http://localhost:8000/chat", null, {
//         params: { query: message },
//       });

//       console.log("Response from backend:", response.data);

//       if (response.data && response.data.response) {
//         const botReply = response.data.response;
//         setMessages((prevMessages) => [
//           ...prevMessages,
//           { sender: "ChatBot", text: botReply },
//         ]);
//       } else {
//         throw new Error("Invalid response format");
//       }
//     } catch (error) {
//       console.error("Error calling the backend:", error);
//       setMessages((prevMessages) => [
//         ...prevMessages,
//         { sender: "ChatBot", text: "Something went wrong. Please try again." },
//       ]);
//     } finally {
//       setIsLoading(false); // Hide the "Generating response..." message
//     }
//   };

//   const toggleSidebar = () => {
//     setIsSidebarVisible(!isSidebarVisible); // Toggle sidebar visibility
//   };

//   const addNewChat = () => {
//     const newChatTitle = `Chat ${chats.length + 1}`;
//     setChats([...chats, newChatTitle]);
//     setActiveChat(newChatTitle); // Automatically switch to the new chat
//     setMessages([]); // Clear any existing messages for the new chat
//     setFirstMessageSent(false); // Reset the first message flag
//   };

//   const switchChat = (chat) => {
//     setActiveChat(chat);
//     setMessages([]); // Reset messages for simplicity
//     setFirstMessageSent(false); // Reset the first message flag
//   };

//   const deleteChat = (chatToDelete) => {
//     const updatedChats = chats.filter((chat) => chat !== chatToDelete);
//     setChats(updatedChats);

//     // If the deleted chat is active, reset the active chat
//     if (chatToDelete === activeChat) {
//       setActiveChat(null);
//       setMessages([]);
//       setFirstMessageSent(false);
//     }
//   };

//   return (
//     <div className="flex h-screen bg-gray-900 text-white">
//       {/* Sidebar */}
//       <div
//         className={`transition-all duration-500 ease-in-out fixed top-0 left-0 h-full bg-gray-800 w-64 shadow-lg z-10 transform ${
//           isSidebarVisible ? "translate-x-0" : "-translate-x-[90%]"
//         }`}
//         onClick={() => {
//           if (!isSidebarVisible) {
//             setIsSidebarVisible(true); // Expand if the sidebar is currently hidden
//           }
//         }} // Sidebar expands if clicked when hidden
//       >
//         <Sidebar
//           chats={chats}
//           activeChat={activeChat}
//           onAddNewChat={addNewChat}
//           onSwitchChat={switchChat}
//           onDeleteChat={deleteChat}
//           onToggleSidebar={toggleSidebar} // Passing toggleSidebar function
//         />
//       </div>
//       {/* Chat Area */}
//       <div
//         className={`flex-1 transition-all duration-500 ${
//           isSidebarVisible ? "ml-64" : "ml-10"
//         }`} // Remove margin when sidebar is hidden
//       >
//         <div className="flex flex-col h-full">
//           {/* ChatBox Display */}
//           <ChatBox messages={messages} isLoading={isLoading} />
//           {/* Message Input */}
//           <MessageInput onSend={handleSend} />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;


import React, { useState, useEffect } from "react";
import axios from "axios";
import Sidebar from "./components/Sidebar";
import ChatBox from "./components/ChatBox";
import MessageInput from "./components/MessageInput";

function App() {
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [chats, setChats] = useState([]); // Stores chat titles dynamically
  const [activeChat, setActiveChat] = useState(null); // Track the current active chat
  const [messages, setMessages] = useState([]); // Messages for the active chat
  // eslint-disable-next-line 
  const [firstMessageSent, setFirstMessageSent] = useState(false); // Track if the first message has been sent
  const [isLoading, setIsLoading] = useState(false); // Track loading state

  // Load data from session storage for each chat
  useEffect(() => {
    const savedChats = JSON.parse(sessionStorage.getItem("chats"));
    if (savedChats) {
      setChats(savedChats);
    }
  }, []);

  // Retrieve messages from sessionStorage for the active chat
  useEffect(() => {
    if (activeChat) {
      const savedMessages = JSON.parse(sessionStorage.getItem(activeChat));
      if (savedMessages) {
        setMessages(savedMessages);
      }
    }
  }, [activeChat]);

  // Save data to sessionStorage when chats or messages change
  useEffect(() => {
    sessionStorage.setItem("chats", JSON.stringify(chats));
    if (activeChat) {
      sessionStorage.setItem(activeChat, JSON.stringify(messages));
    }
  }, [chats, activeChat, messages]);

  const handleSend = async (message) => {
    const updatedMessages = [...messages, { sender: "User", text: message }];
    setMessages(updatedMessages);

    setIsLoading(true); // Show "Generating response..." message

    try {
      const response = await axios.post("http://localhost:8000/chat", null, {
        params: { query: message },
      });

      console.log("Response from backend:", response.data);

      if (response.data && response.data.response) {
        const botReply = response.data.response;
        setMessages((prevMessages) => [
          ...prevMessages,
          { sender: "ChatBot", text: botReply },
        ]);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error("Error calling the backend:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: "ChatBot", text: "Something went wrong. Please try again." },
      ]);
    } finally {
      setIsLoading(false); // Hide the "Generating response..." message
    }
  };

  const toggleSidebar = () => {
    setIsSidebarVisible(!isSidebarVisible); // Toggle sidebar visibility
  };

  const addNewChat = () => {
    const newChatTitle = `Chat ${chats.length + 1}`;
    setChats([...chats, newChatTitle]);
    setActiveChat(newChatTitle); // Automatically switch to the new chat
    setMessages([]); // Clear any existing messages for the new chat
    setFirstMessageSent(false); // Reset the first message flag
  };

  const switchChat = (chat) => {
    setActiveChat(chat);
    setMessages([]); // Reset messages for simplicity
    setFirstMessageSent(false); // Reset the first message flag
  };

  const deleteChat = (chatToDelete) => {
    const updatedChats = chats.filter((chat) => chat !== chatToDelete);
    setChats(updatedChats);

    // If the deleted chat is active, reset the active chat
    if (chatToDelete === activeChat) {
      setActiveChat(null);
      setMessages([]);
      setFirstMessageSent(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div
        className={`transition-all duration-500 ease-in-out fixed top-0 left-0 h-full bg-gray-800 w-64 shadow-lg z-10 transform ${
          isSidebarVisible ? "translate-x-0" : "-translate-x-[90%]"
        }`}
        onClick={() => {
          if (!isSidebarVisible) {
            setIsSidebarVisible(true); // Expand if the sidebar is currently hidden
          }
        }} // Sidebar expands if clicked when hidden
      >
        <Sidebar
          chats={chats}
          activeChat={activeChat}
          onAddNewChat={addNewChat}
          onSwitchChat={switchChat}
          onDeleteChat={deleteChat}
          onToggleSidebar={toggleSidebar} // Passing toggleSidebar function
        />
      </div>
      {/* Chat Area */}
      <div
        className={`flex-1 transition-all duration-500 ${
          isSidebarVisible ? "ml-64" : "ml-10"
        }`} // Remove margin when sidebar is hidden
      >
        <div className="flex flex-col h-full">
          {/* ChatBox Display */}
          <ChatBox messages={messages} isLoading={isLoading} />
          {/* Message Input */}
          <MessageInput onSend={handleSend} isSending={isLoading} />
        </div>
      </div>
    </div>
  );
}

export default App;
