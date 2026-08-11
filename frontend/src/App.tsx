import { useState } from 'react'

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full p-6 bg-card border rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold mb-4 text-primary">EXtendQuality</h1>
        <p className="text-muted-foreground mb-6">
          AI-assisted Industrial Quality Intelligence Platform.
        </p>
        <div className="flex gap-4">
          <div className="p-4 border rounded bg-secondary flex-1 text-center">
            <div className="text-sm text-muted-foreground">Frontend Status</div>
            <div className="font-bold text-green-500">Online</div>
          </div>
          <div className="p-4 border rounded bg-secondary flex-1 text-center">
            <div className="text-sm text-muted-foreground">Tailwind CSS</div>
            <div className="font-bold text-green-500">Active</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
