import './App.css'
import { HeroUIProvider } from '@heroui/react'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Chatbot from "./components/Chatbot"
import Home from "./components/Home";

function App() {
  return (
    <HeroUIProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={
            <div className="chatbot">
              <Chatbot />
            </div>
          } />
        </Routes>
      </Router>
    </HeroUIProvider>
  )
}

export default App