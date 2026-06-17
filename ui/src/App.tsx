
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from '@/components/ui/toaster'
import { ThemeProvider } from '@/components/theme-provider'
import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'
import { Dashboard } from '@/pages/dashboard'
import { Configuration } from '@/pages/configuration'
import { Indicators } from '@/pages/indicators'
import { Instruments } from '@/pages/instruments'
import { Models } from '@/pages/models'
import { Calendar } from '@/pages/calendar'
import { Monitoring } from '@/pages/monitoring'
import { DataManagement } from '@/pages/data-management'
import { Charts } from '@/pages/charts'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="system" storageKey="algo-trade-theme">
        <Router>
          <div className="flex h-screen bg-background">
            <Sidebar />
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-x-hidden overflow-y-auto">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/configuration" element={<Configuration />} />
                  <Route path="/indicators" element={<Indicators />} />
                  <Route path="/instruments" element={<Instruments />} />
                  <Route path="/models" element={<Models />} />
                  <Route path="/calendar" element={<Calendar />} />
                  <Route path="/monitoring" element={<Monitoring />} />
                  <Route path="/data-management" element={<DataManagement />} />
                  <Route path="/charts" element={<Charts />} />
                </Routes>
              </main>
            </div>
          </div>
          <Toaster />
        </Router>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App