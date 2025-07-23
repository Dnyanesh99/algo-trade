import { useState, useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { toast } from '@/hooks/use-toast'
import {
  Search,
  TrendingUp,
  Activity,
  BarChart3,
  RefreshCw,
  Calendar,
  Clock
} from 'lucide-react'

interface Instrument {
  instrument_id: number
  tradingsymbol: string
  name: string
  exchange: string
  segment: string
  instrument_type: string
  last_price: number | null
}

interface OHLCVData {
  ts: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  oi?: number
}

interface ChartDataResponse {
  success: boolean
  data: OHLCVData[]
  total_records: number
  timeframe: string
  instrument_id: number
  start_time?: string
  end_time?: string
}

const TIMEFRAMES = [
  { value: '1min', label: '1 Minute', shortLabel: '1m' },
  { value: '5min', label: '5 Minutes', shortLabel: '5m' },
  { value: '15min', label: '15 Minutes', shortLabel: '15m' },
  { value: '60min', label: '1 Hour', shortLabel: '1h' }
]

const CHART_INTERVALS = [
  { value: '1000', label: 'Last 1000 candles' },
  { value: '500', label: 'Last 500 candles' },
  { value: '250', label: 'Last 250 candles' },
  { value: '100', label: 'Last 100 candles' }
]

export function Charts() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedInstrument, setSelectedInstrument] = useState<Instrument | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState('15min')
  const [chartInterval, setChartInterval] = useState('1000')
  const [showSearch, setShowSearch] = useState(true)
  // const [isLoading, setIsLoading] = useState(false)

  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const intervalRef = useRef<number | null>(null)

  // Search instruments
  const { data: instrumentsData, isLoading: searchLoading } = useQuery({
    queryKey: ['instruments-search', searchTerm],
    queryFn: async () => {
      if (!searchTerm || searchTerm.length < 2) return { instruments: [], total_found: 0 }

      const response = await fetch(`/api/charts/instruments/search?q=${encodeURIComponent(searchTerm)}`)
      if (!response.ok) throw new Error('Failed to search instruments')
      return response.json()
    },
    enabled: searchTerm.length >= 2,
    staleTime: 30000 // Cache for 30 seconds
  })

  // Get available timeframes for selected instrument
  const { data: timeframesData } = useQuery({
    queryKey: ['timeframes', selectedInstrument?.instrument_id],
    queryFn: async () => {
      if (!selectedInstrument) return { timeframes: [] }

      const response = await fetch(`/api/charts/instruments/${selectedInstrument.instrument_id}/timeframes`)
      if (!response.ok) throw new Error('Failed to get timeframes')
      return response.json()
    },
    enabled: !!selectedInstrument
  })

  // Get chart data
  const { data: chartData, refetch: refetchChartData } = useQuery({
    queryKey: ['chart-data', selectedInstrument?.instrument_id, selectedTimeframe, chartInterval],
    queryFn: async (): Promise<ChartDataResponse> => {
      if (!selectedInstrument) throw new Error('No instrument selected')

      const response = await fetch(
        `/api/charts/data/${selectedInstrument.instrument_id}/${selectedTimeframe}?limit=${chartInterval}`
      )
      if (!response.ok) throw new Error('Failed to get chart data')
      return response.json()
    },
    enabled: !!selectedInstrument,
    staleTime: 10000 // Cache for 10 seconds
  })

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: 'transparent' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#485563',
      },
      timeScale: {
        borderColor: '#485563',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // @ts-ignore
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    chartRef.current = chart
    seriesRef.current = candlestickSeries

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
      }
    }
  }, [])

  // Update chart data
  useEffect(() => {
    if (!chartData || !seriesRef.current) return

    const formattedData: CandlestickData[] = chartData.data.map((item) => ({
      time: (new Date(item.ts).getTime() / 1000) as Time,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close,
    }))

    seriesRef.current.setData(formattedData)
  }, [chartData])

  // Auto-refresh for real-time updates
  useEffect(() => {
    if (!selectedInstrument || selectedTimeframe !== '1min') return

    intervalRef.current = setInterval(() => {
      refetchChartData()
    }, 60000) // Refresh every minute for 1min timeframe

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [selectedInstrument, selectedTimeframe, refetchChartData])

  const handleInstrumentSelect = (instrument: Instrument) => {
    setSelectedInstrument(instrument)
    setShowSearch(false)
    setSearchTerm('')
  }

  const handleTimeframeChange = (timeframe: string) => {
    setSelectedTimeframe(timeframe)
  }

  const handleRefresh = () => {
    if (selectedInstrument) {
      refetchChartData()
      toast({
        title: 'Chart Refreshed',
        description: `Updated chart data for ${selectedInstrument.tradingsymbol}`,
      })
    }
  }

  const formatPrice = (price: number | null) => {
    if (price === null) return 'N/A'
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(price)
  }

  const getAvailableTimeframes = () => {
    if (!timeframesData || !timeframesData.timeframes) return []
    return TIMEFRAMES.filter(tf => timeframesData.timeframes.includes(tf.value))
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BarChart3 className="h-8 w-8" />
            Trading Charts
          </h1>
          <p className="text-muted-foreground mt-1">
            Real-time charts with multiple timeframes
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSearch(true)}
            className="flex items-center gap-2"
          >
            <Search className="h-4 w-4" />
            Search Instruments
          </Button>
          {selectedInstrument && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          )}
        </div>
      </div>

      {/* Search Panel */}
      {showSearch && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Search Instruments
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="search">Search by symbol or name</Label>
              <Input
                id="search"
                placeholder="e.g., NIFTY, BANKNIFTY, SBIN..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="max-w-md"
              />
            </div>

            {searchLoading && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <RefreshCw className="h-4 w-4 animate-spin" />
                Searching...
              </div>
            )}

            {instrumentsData && instrumentsData.instruments.length > 0 && (
              <div className="grid gap-2 max-h-64 overflow-y-auto">
                {instrumentsData.instruments.map((instrument: Instrument) => (
                  <div
                    key={instrument.instrument_id}
                    className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent cursor-pointer"
                    onClick={() => handleInstrumentSelect(instrument)}
                  >
                    <div className="flex items-center gap-3">
                      <div>
                        <div className="font-medium">{instrument.tradingsymbol}</div>
                        <div className="text-sm text-muted-foreground">
                          {instrument.name} • {instrument.exchange}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{instrument.segment}</Badge>
                      {instrument.last_price && (
                        <div className="text-sm font-mono">
                          {formatPrice(instrument.last_price)}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {searchTerm.length >= 2 && instrumentsData && instrumentsData.instruments.length === 0 && !searchLoading && (
              <div className="text-center py-8 text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <div>No instruments found for "{searchTerm}"</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Chart Panel */}
      {selectedInstrument && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                {selectedInstrument.tradingsymbol}
                <Badge variant="outline">{selectedInstrument.exchange}</Badge>
              </CardTitle>
              <div className="flex items-center gap-2">
                <Select value={selectedTimeframe} onValueChange={handleTimeframeChange}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {getAvailableTimeframes().map((tf) => (
                      <SelectItem key={tf.value} value={tf.value}>
                        {tf.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={chartInterval} onValueChange={setChartInterval}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CHART_INTERVALS.map((interval) => (
                      <SelectItem key={interval.value} value={interval.value}>
                        {interval.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Chart Stats */}
              {chartData && (
                <div className="flex items-center gap-6 text-sm text-muted-foreground">
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    {chartData.total_records} candles
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {selectedTimeframe} timeframe
                  </div>
                  {selectedInstrument.last_price && (
                    <div className="flex items-center gap-1">
                      <TrendingUp className="h-4 w-4" />
                      {formatPrice(selectedInstrument.last_price)}
                    </div>
                  )}
                </div>
              )}

              {/* Chart Container */}
              <div
                ref={chartContainerRef}
                className="w-full h-[500px] bg-card rounded-lg border"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Welcome Message */}
      {!selectedInstrument && !showSearch && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BarChart3 className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Welcome to Trading Charts</h3>
            <p className="text-muted-foreground text-center max-w-md mb-6">
              Search for any instrument to view its real-time charts with multiple timeframes.
              Support for 1min, 5min, 15min, and 60min intervals.
            </p>
            <Button onClick={() => setShowSearch(true)} className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Search Instruments
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}