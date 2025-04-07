import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Input } from "@heroui/react";

export default function Home() {
  const [input, setInput] = useState("");
  const navigate = useNavigate();

  const handleSubmit = () => {
    if (input.trim()) {
      navigate("/chat", {
        state: {
          initialMessages: [
            { 
              text: input, 
              sender: "user" 
            }
          ]
        }
      });
    }
  };

  return (
    <div className="max-h-screen flex items-center justify-center bg-gray-50">
      <div className="w-full max-w-md p-8 bg-white rounded-xl shadow-lg">
        <h1 className="text-3xl font-bold text-center mb-8">Welcome to Chatbot</h1>
        <Input
          label="Ask anything you want"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          radius="sm"
          className="mb-4"
        />
        <p className="text-sm text-gray-500 text-center">
          Press ENTER to start
        </p>
      </div>
    </div>
  );
}