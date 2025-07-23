import { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { toast } from '@/hooks/use-toast'
import {
  Upload,
  Download,
  Plus,
  Trash2,
  Edit,
  Filter,
  RefreshCw,
  CheckCircle,
  AlertCircle,
} from 'lucide-react'

interface Instrument {
  id: number;
  label: string;
  tradingsymbol: string;
  exchange: string;
  instrument_type: string;
  segment: string;
  enabled: boolean;
  last_price: number;
  volume: number;
  last_updated: string;
}

const SAMPLE_INSTRUMENTS: Instrument[] = [
  {
    id: 1,
    label: 'NIFTY 50',
    tradingsymbol: 'NIFTY 50',
    exchange: 'NSE',
    instrument_type: 'EQ',
    segment: 'INDICES',
    enabled: true,
    last_price: 19450.25,
    volume: 1000000,
    last_updated: '2024-01-15T14:30:00Z',
  },
  {
    id: 2,
    label: 'SENSEX',
    tradingsymbol: 'SENSEX',
    exchange: 'BSE',
    instrument_type: 'EQ',
    segment: 'INDICES',
    enabled: false,
    last_price: 65432.10,
    volume: 800000,
    last_updated: '2024-01-15T14:30:00Z',
  },
  {
    id: 3,
    label: 'NIFTY BANK',
    tradingsymbol: 'NIFTY BANK',
    exchange: 'NSE',
    instrument_type: 'EQ',
    segment: 'INDICES',
    enabled: false,
    last_price: 43210.75,
    volume: 500000,
    last_updated: '2024-01-15T14:30:00Z',
  },
]

const SEGMENT_TYPES = [
  { value: 'INDICES', label: 'Market Indices' },
  { value: 'EQ', label: 'Equity - Cash Market' },
  { value: 'NFO-OPT', label: 'NSE Options' },
  { value: 'NFO-FUT', label: 'NSE Futures' },
  { value: 'BFO-FUT', label: 'BSE Futures' },
  { value: 'BFO-OPT', label: 'BSE Options' },
]

export function Instruments() {
  const [instruments, setInstruments] = useState<Instrument[]>(SAMPLE_INSTRUMENTS)
  const [filteredInstruments, setFilteredInstruments] = useState<Instrument[]>(SAMPLE_INSTRUMENTS)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedSegment, setSelectedSegment] = useState<string>('all')
  const [selectedExchange, setSelectedExchange] = useState<string>('all')
  const [showEnabledOnly, setShowEnabledOnly] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [editingInstrument, setEditingInstrument] = useState<Instrument | null>(null)

  // Apply filters whenever dependencies change
  useEffect(() => {
    const filterInstruments = () => {
      let filtered = instruments

      if (searchTerm) {
        filtered = filtered.filter(instrument =>
          instrument.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
          instrument.tradingsymbol.toLowerCase().includes(searchTerm.toLowerCase())
        )
      }

      if (selectedSegment !== 'all') {
        filtered = filtered.filter(instrument => instrument.segment === selectedSegment)
      }

      if (selectedExchange !== 'all') {
        filtered = filtered.filter(instrument => instrument.exchange === selectedExchange)
      }

      if (showEnabledOnly) {
        filtered = filtered.filter(instrument => instrument.enabled)
      }

      setFilteredInstruments(filtered)
    }

    filterInstruments()
  }, [searchTerm, selectedSegment, selectedExchange, showEnabledOnly, instruments])

  const toggleInstrument = (id: number) => {
    setInstruments(prev => prev.map(instrument =>
      instrument.id === id ? { ...instrument, enabled: !instrument.enabled } : instrument
    ))
  }

  const deleteInstrument = (id: number) => {
    setInstruments(prev => prev.filter(instrument => instrument.id !== id))
    toast({
      title: 'Instrument Deleted',
      description: 'The instrument has been removed from the list.',
    })
  }

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
        const requiredColumns = ['label', 'tradingsymbol', 'exchange', 'instrument_type', 'segment']
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
        const newInstruments = lines.slice(1)
          .filter(line => line.trim())
          .map((line, index) => {
            const values = line.split(',').map(v => v.trim())
            const instrument: { [key: string]: string | number | boolean } = {
              id: instruments.length + index + 1,
              enabled: false,
              last_price: 0,
              volume: 0,
              last_updated: new Date().toISOString(),
            }

            headers.forEach((header, i) => {
              instrument[header] = values[i] || ''
            })

            return instrument
          })

        setInstruments(prev => [...prev, ...newInstruments])
        toast({
          title: 'Instruments Uploaded',
          description: `Successfully uploaded ${newInstruments.length} instruments.`,
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

  const exportInstruments = () => {
    const csvContent = [
      'label,tradingsymbol,exchange,instrument_type,segment,enabled,last_price,volume',
      ...instruments.map(instrument =>
        `${instrument.label},${instrument.tradingsymbol},${instrument.exchange},${instrument.instrument_type},${instrument.segment},${instrument.enabled},${instrument.last_price},${instrument.volume}`
      )
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'instruments.csv'
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const addNewInstrument = () => {
    setEditingInstrument({
      id: instruments.length + 1,
      label: '',
      tradingsymbol: '',
      exchange: 'NSE',
      instrument_type: 'EQ',
      segment: 'INDICES',
      enabled: false,
      last_price: 0,
      volume: 0,
      last_updated: new Date().toISOString(),
    })
  }

  const saveInstrument = (instrument: { [key: string]: string | number | boolean }) => {
    if (instrument.id > instruments.length) {
      // New instrument
      setInstruments(prev => [...prev, instrument])
    } else {
      // Edit existing
      setInstruments(prev => prev.map(i => i.id === instrument.id ? instrument : i))
    }
    setEditingInstrument(null)
    toast({
      title: 'Instrument Saved',
      description: 'The instrument has been saved successfully.',
    })
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Instruments Management</h1>
          <p className="text-muted-foreground mt-1">
            Manage trading instruments and data sources
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
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Upload className="h-4 w-4 mr-2" />
            )}
            Import CSV
          </Button>
          <Button variant="outline" onClick={exportInstruments}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button onClick={addNewInstrument}>
            <Plus className="h-4 w-4 mr-2" />
            Add Instrument
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Filter className="h-5 w-5" />
            <span>Filters</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="search">Search</Label>
              <Input
                id="search"
                placeholder="Search instruments..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="segment">Segment</Label>
              <Select value={selectedSegment} onValueChange={setSelectedSegment}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Segments</SelectItem>
                  {SEGMENT_TYPES.map(segment => (
                    <SelectItem key={segment.value} value={segment.value}>
                      {segment.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="exchange">Exchange</Label>
              <Select value={selectedExchange} onValueChange={setSelectedExchange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Exchanges</SelectItem>
                  <SelectItem value="NSE">NSE</SelectItem>
                  <SelectItem value="BSE">BSE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="enabled-only">Show Enabled Only</Label>
              <div className="flex items-center space-x-2">
                <Switch
                  id="enabled-only"
                  checked={showEnabledOnly}
                  onCheckedChange={setShowEnabledOnly}
                />
                <span className="text-sm text-muted-foreground">
                  {showEnabledOnly ? 'Enabled' : 'All'}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Statistics */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Instruments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{instruments.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Enabled</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {instruments.filter(i => i.enabled).length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Exchanges</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {new Set(instruments.map(i => i.exchange)).size}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Segments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {new Set(instruments.map(i => i.segment)).size}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Instruments Table */}
      <Card>
        <CardHeader>
          <CardTitle>Instruments ({filteredInstruments.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredInstruments.map((instrument) => (
              <div key={instrument.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Switch
                      checked={instrument.enabled}
                      onCheckedChange={() => toggleInstrument(instrument.id)}
                    />
                    {instrument.enabled ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-gray-400" />
                    )}
                  </div>

                  <div>
                    <div className="font-medium">{instrument.label}</div>
                    <div className="text-sm text-muted-foreground">
                      {instrument.tradingsymbol} • {instrument.exchange}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="font-medium">₹{instrument.last_price.toLocaleString()}</div>
                    <div className="text-sm text-muted-foreground">
                      Vol: {instrument.volume.toLocaleString()}
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">{instrument.segment}</Badge>
                    <Badge variant="secondary">{instrument.instrument_type}</Badge>
                  </div>

                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setEditingInstrument(instrument)}
                    >
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteInstrument(instrument.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Edit Instrument Modal */}
      {editingInstrument && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>
                {editingInstrument && editingInstrument.id > instruments.length ? 'Add New Instrument' : 'Edit Instrument'}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="edit-label">Label</Label>
                <Input
                  id="edit-label"
                  value={editingInstrument?.label || ''}
                  onChange={(e) => setEditingInstrument((prev) => prev ? ({ ...prev, label: e.target.value }) : null)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="edit-tradingsymbol">Trading Symbol</Label>
                <Input
                  id="edit-tradingsymbol"
                  value={editingInstrument?.tradingsymbol || ''}
                  onChange={(e) => setEditingInstrument((prev) => prev ? ({ ...prev, tradingsymbol: e.target.value }) : null)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="edit-exchange">Exchange</Label>
                <Select
                  value={editingInstrument?.exchange || ''}
                  onValueChange={(value) => setEditingInstrument((prev) => prev ? ({ ...prev, exchange: value }) : null)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="NSE">NSE</SelectItem>
                    <SelectItem value="BSE">BSE</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="edit-segment">Segment</Label>
                <Select
                  value={editingInstrument?.segment || ''}
                  onValueChange={(value) => setEditingInstrument((prev) => prev ? ({ ...prev, segment: value }) : null)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SEGMENT_TYPES.map(segment => (
                      <SelectItem key={segment.value} value={segment.value}>
                        {segment.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  checked={editingInstrument?.enabled || false}
                  onCheckedChange={(checked) => setEditingInstrument((prev) => prev ? ({ ...prev, enabled: checked }) : null)}
                />
                <Label>Enable for Trading</Label>
              </div>
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setEditingInstrument(null)}>
                  Cancel
                </Button>
                <Button onClick={() => saveInstrument(editingInstrument)}>
                  Save
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}