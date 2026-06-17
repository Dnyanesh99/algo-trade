import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { toast } from '@/hooks/use-toast'
import {
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  Settings,
  Plus,
  Trash2,
  RefreshCw,
  Save,
} from 'lucide-react'

const INDICATOR_CATEGORIES = {
  trend: {
    name: 'Trend Following',
    icon: TrendingUp,
    color: 'text-blue-500',
    indicators: ['MACD', 'ADX', 'SAR', 'AROON', 'HT_TRENDLINE']
  },
  momentum: {
    name: 'Momentum',
    icon: Activity,
    color: 'text-green-500',
    indicators: ['RSI', 'STOCH', 'WILLR', 'CMO', 'BOP']
  },
  volatility: {
    name: 'Volatility',
    icon: BarChart3,
    color: 'text-orange-500',
    indicators: ['ATR', 'BBANDS', 'STDDEV']
  },
  volume: {
    name: 'Volume',
    icon: TrendingDown,
    color: 'text-purple-500',
    indicators: ['OBV', 'ADOSC']
  },
  cycle: {
    name: 'Cycle',
    icon: RefreshCw,
    color: 'text-indigo-500',
    indicators: ['HT_DCPERIOD', 'HT_PHASOR']
  },
  other: {
    name: 'Other',
    icon: Settings,
    color: 'text-gray-500',
    indicators: ['ULTOSC', 'TRANGE', 'CCI']
  }
}

const SAMPLE_INDICATORS = [
  {
    name: 'macd',
    function: 'MACD',
    category: 'trend',
    enabled: true,
    params: {
      default: { fastperiod: 12, slowperiod: 26, signalperiod: 9 },
      '60m': { fastperiod: 20, slowperiod: 40, signalperiod: 9 }
    }
  },
  {
    name: 'rsi',
    function: 'RSI',
    category: 'momentum',
    enabled: true,
    params: {
      default: { timeperiod: 14 },
      '60m': { timeperiod: 21 }
    }
  },
  {
    name: 'bbands',
    function: 'BBANDS',
    category: 'volatility',
    enabled: true,
    params: {
      default: { timeperiod: 20, nbdevup: 2.0, nbdevdn: 2.0 },
      '60m': { nbdevup: 2.5, nbdevdn: 2.5 }
    }
  },
  {
    name: 'atr',
    function: 'ATR',
    category: 'volatility',
    enabled: true,
    params: {
      default: { timeperiod: 14 }
    }
  },
  {
    name: 'adx',
    function: 'ADX',
    category: 'trend',
    enabled: false,
    params: {
      default: { timeperiod: 14 },
      '60m': { timeperiod: 20 }
    }
  }
]

export function Indicators() {
  const [indicators, setIndicators] = useState(SAMPLE_INDICATORS)
  const [selectedTimeframe, setSelectedTimeframe] = useState('default')
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  const filteredIndicators = indicators.filter(indicator => {
    const matchesSearch = indicator.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         indicator.function.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = !selectedCategory || indicator.category === selectedCategory
    return matchesSearch && matchesCategory
  })

  const toggleIndicator = (name: string) => {
    setIndicators(prev => prev.map(indicator => 
      indicator.name === name ? { ...indicator, enabled: !indicator.enabled } : indicator
    ))
  }

  const updateIndicatorParam = (name: string, timeframe: string, paramKey: string, value: number | string) => {
    setIndicators(prev => prev.map(indicator => {
      if (indicator.name === name) {
        const newParams = { ...indicator.params }
        if (!newParams[timeframe]) {
          newParams[timeframe] = {}
        }
        newParams[timeframe][paramKey] = value
        return { ...indicator, params: newParams }
      }
      return indicator
    }))
  }

  const addTimeframe = (name: string, timeframe: string) => {
    setIndicators(prev => prev.map(indicator => {
      if (indicator.name === name) {
        const newParams = { ...indicator.params }
        newParams[timeframe] = { ...newParams.default }
        return { ...indicator, params: newParams }
      }
      return indicator
    }))
  }

  const removeTimeframe = (name: string, timeframe: string) => {
    if (timeframe === 'default') return
    setIndicators(prev => prev.map(indicator => {
      if (indicator.name === name) {
        const newParams = { ...indicator.params }
        delete newParams[timeframe]
        return { ...indicator, params: newParams }
      }
      return indicator
    }))
  }

  const saveConfiguration = () => {
    // Here you would send the configuration to your backend
    toast({
      title: 'Configuration Saved',
      description: 'Indicator configuration has been saved successfully.',
    })
  }

  const resetToDefaults = () => {
    toast({
      title: 'Reset to Defaults',
      description: 'All indicator configurations have been reset to default values.',
    })
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Technical Indicators</h1>
          <p className="text-muted-foreground mt-1">
            Configure technical indicators for signal generation
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={resetToDefaults}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button onClick={saveConfiguration}>
            <Save className="h-4 w-4 mr-2" />
            Save Configuration
          </Button>
        </div>
      </div>

      {/* Categories and Search */}
      <div className="flex flex-wrap gap-2">
        <Button
          variant={selectedCategory === null ? 'default' : 'outline'}
          size="sm"
          onClick={() => setSelectedCategory(null)}
        >
          All Categories
        </Button>
        {Object.entries(INDICATOR_CATEGORIES).map(([key, category]) => {
          const Icon = category.icon
          return (
            <Button
              key={key}
              variant={selectedCategory === key ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory(key)}
            >
              <Icon className={`h-4 w-4 mr-2 ${category.color}`} />
              {category.name}
            </Button>
          )
        })}
      </div>

      <div className="flex items-center space-x-4">
        <div className="flex-1">
          <Input
            placeholder="Search indicators..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="max-w-sm"
          />
        </div>
        <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">Default</SelectItem>
            <SelectItem value="5m">5 min</SelectItem>
            <SelectItem value="15m">15 min</SelectItem>
            <SelectItem value="60m">60 min</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Indicators Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredIndicators.map((indicator) => {
          const category = INDICATOR_CATEGORIES[indicator.category as keyof typeof INDICATOR_CATEGORIES]
          const CategoryIcon = category.icon
          const currentParams = indicator.params[selectedTimeframe] || indicator.params.default
          
          return (
            <Card key={indicator.name} className={`${indicator.enabled ? 'ring-2 ring-primary/20' : ''}`}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <CategoryIcon className={`h-4 w-4 ${category.color}`} />
                    <CardTitle className="text-lg">{indicator.function}</CardTitle>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="text-xs">
                      {category.name}
                    </Badge>
                    <Switch
                      checked={indicator.enabled}
                      onCheckedChange={() => toggleIndicator(indicator.name)}
                    />
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {/* Timeframe Management */}
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Timeframe: {selectedTimeframe}</Label>
                  <div className="flex items-center space-x-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        const newTimeframe = prompt('Enter timeframe (e.g., 30m, 1h):')
                        if (newTimeframe) {
                          addTimeframe(indicator.name, newTimeframe)
                        }
                      }}
                    >
                      <Plus className="h-3 w-3" />
                    </Button>
                    {selectedTimeframe !== 'default' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeTimeframe(indicator.name, selectedTimeframe)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                </div>

                {/* Parameters */}
                <div className="space-y-3">
                  {Object.entries(currentParams).map(([paramKey, paramValue]) => (
                    <div key={paramKey} className="flex items-center space-x-2">
                      <Label className="text-xs min-w-[80px]">{paramKey}</Label>
                      <Input
                        type="number"
                        value={paramValue as string | number}
                        onChange={(e) => updateIndicatorParam(
                          indicator.name,
                          selectedTimeframe,
                          paramKey,
                          paramKey.includes('period') ? parseInt(e.target.value) : parseFloat(e.target.value)
                        )}
                        className="h-8 text-sm"
                        step={paramKey.includes('period') ? '1' : '0.1'}
                      />
                    </div>
                  ))}
                </div>

                {/* Available Timeframes */}
                <div className="flex flex-wrap gap-1">
                  {Object.keys(indicator.params).map(timeframe => (
                    <Badge key={timeframe} variant="secondary" className="text-xs">
                      {timeframe}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Configuration Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {indicators.filter(i => i.enabled).length}
              </div>
              <p className="text-sm text-muted-foreground">Enabled Indicators</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {indicators.length}
              </div>
              <p className="text-sm text-muted-foreground">Total Available</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {Object.keys(INDICATOR_CATEGORIES).length}
              </div>
              <p className="text-sm text-muted-foreground">Categories</p>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {indicators.reduce((acc, indicator) => acc + Object.keys(indicator.params).length, 0)}
              </div>
              <p className="text-sm text-muted-foreground">Timeframe Configs</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}