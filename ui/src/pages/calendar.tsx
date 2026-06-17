import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from '@/hooks/use-toast'
import {
  Calendar as CalendarIcon,
  Upload,
  Download,
  Plus,
  Trash2,
  Clock,
} from 'lucide-react'

const SAMPLE_HOLIDAYS = [
  { date: '2024-01-26', name: 'Republic Day', type: 'national' },
  { date: '2024-03-08', name: 'Mahashivratri', type: 'religious' },
  { date: '2024-03-25', name: 'Holi', type: 'religious' },
  { date: '2024-08-15', name: 'Independence Day', type: 'national' },
  { date: '2024-10-02', name: 'Gandhi Jayanti', type: 'national' },
  { date: '2024-11-01', name: 'Diwali', type: 'religious' },
  { date: '2024-12-25', name: 'Christmas', type: 'religious' },
]

const SAMPLE_SESSIONS = [
  { date: '2024-11-01', name: 'Diwali Muhurat Trading', open: '18:00', close: '19:15' },
  { date: '2024-02-01', name: 'Budget Day', open: '09:15', close: '12:30' },
]

export function Calendar() {
  const [holidays, setHolidays] = useState(SAMPLE_HOLIDAYS)
  const [sessions] = useState(SAMPLE_SESSIONS)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      toast({
        title: 'Invalid File Type',
        description: 'Please upload a CSV file.',
        variant: 'destructive',
      })
      return
    }

    setUploading(true)
    const reader = new FileReader()
    
    reader.onload = (e) => {
      try {
        const csvData = e.target?.result as string
        const lines = csvData.split('\n')
        const headers = lines[0].split(',').map(h => h.trim())
        
        // Validate required columns
        const requiredColumns = ['date', 'name', 'type']
        const missingColumns = requiredColumns.filter(col => !headers.includes(col))
        
        if (missingColumns.length > 0) {
          toast({
            title: 'Invalid CSV Format',
            description: `Missing required columns: ${missingColumns.join(', ')}`,
            variant: 'destructive',
          })
          setUploading(false)
          return
        }

        // Parse CSV data
        const newHolidays = lines.slice(1)
          .filter(line => line.trim())
          .map((line) => {
            const values = line.split(',').map(v => v.trim())
            const holiday: Record<string, string> = {}
            
            headers.forEach((header, i) => {
              holiday[header] = values[i] || ''
            })
            
            return holiday
          })

        setHolidays(prev => [...prev, ...newHolidays])
        toast({
          title: 'Holidays Uploaded',
          description: `Successfully uploaded ${newHolidays.length} holidays.`,
        })
        
        // Clear the file input
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      } catch (error) {
        toast({
          title: 'Upload Failed',
          description: 'Failed to parse CSV file. Please check the format.',
          variant: 'destructive',
        })
      } finally {
        setUploading(false)
      }
    }
    
    reader.readAsText(file)
  }

  const exportHolidays = () => {
    const csvContent = [
      'date,name,type',
      ...holidays.map(holiday => `${holiday.date},${holiday.name},${holiday.type}`)
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'holidays.csv'
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const deleteHoliday = (date: string) => {
    setHolidays(prev => prev.filter(holiday => holiday.date !== date))
    toast({
      title: 'Holiday Deleted',
      description: 'The holiday has been removed from the calendar.',
    })
  }

  const addHoliday = () => {
    const date = prompt('Enter date (YYYY-MM-DD):')
    const name = prompt('Enter holiday name:')
    const type = prompt('Enter type (national/religious/market):')
    
    if (date && name && type) {
      setHolidays(prev => [...prev, { date, name, type }])
      toast({
        title: 'Holiday Added',
        description: 'The holiday has been added to the calendar.',
      })
    }
  }

  const getHolidayTypeColor = (type: string) => {
    switch (type) {
      case 'national': return 'bg-red-100 text-red-800'
      case 'religious': return 'bg-orange-100 text-orange-800'
      case 'market': return 'bg-blue-100 text-blue-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Market Calendar</h1>
          <p className="text-muted-foreground mt-1">
            Manage trading holidays and special sessions
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
          />
          <Button
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? (
              <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full mr-2" />
            ) : (
              <Upload className="h-4 w-4 mr-2" />
            )}
            Import CSV
          </Button>
          <Button variant="outline" onClick={exportHolidays}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button onClick={addHoliday}>
            <Plus className="h-4 w-4 mr-2" />
            Add Holiday
          </Button>
        </div>
      </div>

      <Tabs defaultValue="holidays">
        <TabsList>
          <TabsTrigger value="holidays">Holidays</TabsTrigger>
          <TabsTrigger value="sessions">Special Sessions</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="holidays" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Total Holidays</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{holidays.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">This Month</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-600">
                  {holidays.filter(h => h.date.startsWith('2024-01')).length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Upcoming</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-600">
                  {holidays.filter(h => new Date(h.date) > new Date()).length}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Holiday Calendar</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {holidays.map((holiday, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <CalendarIcon className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-medium">{holiday.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {new Date(holiday.date).toLocaleDateString('en-IN', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge className={getHolidayTypeColor(holiday.type)}>
                        {holiday.type}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteHoliday(holiday.date)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sessions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Special Trading Sessions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {sessions.map((session, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="font-medium">{session.name}</div>
                        <div className="text-sm text-muted-foreground">
                          {new Date(session.date).toLocaleDateString('en-IN')}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <Badge variant="outline">
                        {session.open} - {session.close}
                      </Badge>
                      <Button variant="ghost" size="sm">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Trading Hours</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="market-open">Market Open Time</Label>
                  <Input id="market-open" type="time" defaultValue="09:15" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="market-close">Market Close Time</Label>
                  <Input id="market-close" type="time" defaultValue="15:30" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Timezone Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="timezone">Timezone</Label>
                <Input id="timezone" defaultValue="Asia/Kolkata" />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}