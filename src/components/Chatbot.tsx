import React, { useState, useEffect, useRef } from "react";
import { Button, Input, Card, CardBody } from "@heroui/react";
import { useLocation } from "react-router-dom";

interface Message {
  text: string;
  sender: "user" | "bot";
  retry?: boolean;
  isLoading?: boolean;
  error?: boolean;
  streamId?: string; // 唯一标识符
}

function Chatbot() {
  const location = useLocation();
  const {initialMessages = [] } = location.state || {};
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messageRefs = useRef<HTMLDivElement[]>([]);
  // const [allMessagesCompleted, setAllMessagesCompleted] = useState(false);

  // ================ 消息处理 ================
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const sendMessage = async () => {
    if (input.trim()) {
      const userMessage: Message = {
        text: input,
        sender: "user",
      };
      setMessages((prev) => [...prev, userMessage]);
      setInput("");
      await fetchBotResponse(input);
    }
  };

  const retryMessage = () => {
    // 最后一条chatbot消息
    const lastBotMessage = [...messages]
      .reverse()
      .find(msg => msg.sender === "bot");
  
    if (!lastBotMessage) return;
  
    // 找到触发该回复的用户消息
    const userMessageIndex = messages.indexOf(lastBotMessage) - 1;
    const userMessage = messages[userMessageIndex];
  
    // 移除旧的chatbot消息
    setMessages(prev => 
      prev.filter(msg => msg !== lastBotMessage)
    );
  
    // 重新发送请求
    if (userMessage?.text) {
      fetchBotResponse(userMessage.text);
    }
  };

  const fetchBotResponse = async (userInput: string) => {
    try {
      const streamId = Date.now().toString();
      if (messages.some(msg => msg.isLoading)) return;
      setMessages(prev => [
        ...prev,
        { 
          text: "", 
          sender: "bot", 
          isLoading: true,
          streamId: streamId 
        }
      ]);
  
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userInput }),
      });
  
      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      if (!response.body) throw new Error("Empty response body");
  
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
  
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
  
        buffer += decoder.decode(value, { stream: true });
  
        const separator = buffer.includes("\r\n\r\n") ? "\r\n\r\n" : "\n\n";
        let eventEnd;
  
        while ((eventEnd = buffer.indexOf(separator)) !== -1) {
          const event = buffer.slice(0, eventEnd);
          buffer = buffer.slice(eventEnd + separator.length);
  
          if (!event.startsWith("data: ")) continue;
          const data = event.slice(6).trim();
  
          if (data === "[DONE]") {
            setMessages(prev => 
              prev.map(msg => 
                msg.streamId === streamId 
                  ? { ...msg, isLoading: false, streamId: undefined } 
                  : msg
              )
            );
            break;
          }
  
          // 拼接逻辑
          setMessages(prev => {
            return prev.map(msg => {
              if (msg.streamId === streamId) {
                const newText = msg.text + (msg.text && data && !data.startsWith(" ") ? " " : "") + data;
                return { ...msg, text: newText };
              }
              return msg;
            });
          });
  
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }
    } catch (error) {
      let errorMessage = "请求失败";
      if (error instanceof Error) errorMessage += `: ${error.message}`;
      setMessages((prev) => [
        ...prev,
        { 
          text: errorMessage, 
          sender: "bot", 
          error: true,
          retry: true, 
          streamId: undefined 
        },
      ]);
    }
  };

  // ================ 副作用处理 ================
  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (scrollContainer) {
      requestAnimationFrame(() => {
        scrollContainer.scrollTo({
          top: scrollContainer.scrollHeight,
          behavior: "smooth",
        });
      });
    }
  }, [messages]);

  useEffect(() => {
    // 仅处理初始消息中的用户消息
    const lastUserMessage = initialMessages
      .filter((msg: Message) => msg.sender === "user")
      .pop();
  
    if (lastUserMessage?.text) {
      fetchBotResponse(lastUserMessage.text);
    }
  }, []); // 空依赖数组，仅在挂载时执行一次

  useEffect(() => {
    return () => {
      // 清除任何定时器或资源
    };
  }, []);

  return (
    <div className="flex flex-col h-[600px] w-[1000px] mx-auto border-4 border-gray-200 rounded-xl bg-white shadow-lg p-4">
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto mb-4 space-y-2 border border-gray-100 rounded-lg p-2"
      >
        {messages.map((message, index) => (
          <div
            key={index}
            ref={(el) => {
              if (el) {
                messageRefs.current[index] = el;
              }
            }}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <Card
              className={`max-w-[80%] p-3 rounded-xl transition-all ${
                message.sender === "user"
                  ? "bg-blue-500 text-white rounded-br-none"
                  : "bg-gray-100 text-gray-800 rounded-bl-none"
              }`}
              radius="lg"
              shadow="none"
            >
              <CardBody className="p-0">
                <div className="flex items-center justify-between">
                  <span className="text-sm whitespace-pre-wrap break-words">
                    {message.text}
                  </span>
                  {message.sender === "bot" &&
                    index === (() => {
                      let lastIndex = -1;
                      for (let i = messages.length - 1; i >= 0; i--) {
                        if (messages[i].sender === "bot") {
                          lastIndex = i;
                          break;
                        }
                      }
                      return lastIndex;
                    })() && (
                      <Button
                        variant="ghost"
                        size="sm"
                        color="danger"
                        onPress={retryMessage}
                        className="ml-2 h-6 w-6 p-1 text-gray-600 hover:text-gray-800"
                      >
                        Retry
                      </Button>
                  )}
                </div>
              </CardBody>
            </Card>
          </div>
        ))}
      </div>

      <div className="flex gap-2 items-center mt-4">
        <Input
          label="Ask anything you want"
          value={input}
          onChange={handleInputChange}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          radius="full"
        />
        <Button color="primary" onPress={sendMessage} radius="full">
          Send
        </Button>
      </div>
    </div>
  );
}

export default Chatbot;