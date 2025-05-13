import { useState } from 'react'
import './App.css'
import ImageUpload from './ImageUpload'

function App() {
  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Lung Cancer Detection System</h1>
      </header>
      
      <main className="app-main">
        <ImageUpload />
      </main>
    </div>
  )
}

export default App
