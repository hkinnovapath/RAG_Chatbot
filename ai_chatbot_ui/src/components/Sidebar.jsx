import React from "react";
import { FaTrash, FaTimes } from "react-icons/fa";

function Sidebar({ chats, activeChat, onAddNewChat, onSwitchChat, onDeleteChat, onToggleSidebar }) {
  return (
    <div className="flex flex-col h-full p-4 bg-gray-800 relative transition-all ease-in-out duration-500">
      {/* Header with the cross button beside "Conversations" */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold">Conversation History</h2>
        <button
          onClick={onToggleSidebar}  // This is now connected to the functionality
          className="text-white hover:text-gray-400"
        >
          <FaTimes size={20} /> {/* Cross icon for hiding */}
        </button>
      </div>

      {/* Chat List */}
      <ul className="flex-1 overflow-y-auto space-y-2 custom-scrollbar pr-2">
        {chats.map((chat, index) => (
          <li
            key={index}
            className={`flex items-center justify-between p-2 rounded-lg cursor-pointer ${
              chat === activeChat
                ? "bg-blue-500 text-white"
                : "bg-gray-700 hover:bg-gray-600"
            }`}
          >
            <span onClick={() => onSwitchChat(chat)} className="flex-1">
              {chat}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation(); // Prevent triggering the switchChat on delete click
                onDeleteChat(chat);
              }}
              className="text-gray-800 hover:text-red-600">
              <FaTrash />
            </button>
          </li>
        ))}
      </ul>

      {/* Add New Chat Button */}
      <button
        onClick={onAddNewChat}
        className="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg shadow"
      >
        Add New Chat
      </button>
    </div>
  );
}

export default Sidebar;
