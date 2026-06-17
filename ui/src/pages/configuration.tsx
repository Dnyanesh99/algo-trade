import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { toast } from '@/hooks/use-toast'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Settings, 
  Database, 
  TrendingUp, 
  Shield, 
  Server, 
  AlertCircle, 
  Save, 
  Download, 
  Upload, 
} from 'lucide-react'

const fetchConfig = async () => {
  const response = await fetch('/api/config')
  if (!response.ok) {
    throw new Error('Failed to fetch configuration')
  }
  return response.json()
}

interface ConfigUpdateData {
  section: string;
  data: Record<string, unknown>;
}

const updateConfigSection = async ({ section, data }: ConfigUpdateData) => {
  const response = await fetch(`/api/config/${section}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ data }),
  })
  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Failed to update configuration')
  }
  return response.json()
}

// const deleteConfigKey = async ({ section, key }: { section: string, key: string }) => {
//   const response = await fetch(`/api/config/${section}/${key}`, {
//     method: 'DELETE',
//   })
//   if (!response.ok) {
//     const errorData = await response.json()
//     throw new Error(errorData.detail || 'Failed to delete configuration key')
//   }
//   return response.json()
// }

const renderValueInput = (key: string, value: unknown, section: string, path: string[], handleUpdate: (path: string[], value: unknown) => void) => {
  const fullPath = [...path, key];
  const id = fullPath.join('-');

  if (typeof value === 'boolean') {
    return (
      <Switch
        id={id}
        checked={value}
        onCheckedChange={(checked) => handleUpdate(fullPath, checked)}
      />
    );
  }
  if (typeof value === 'number') {
    return (
      <Input
        id={id}
        type="number"
        value={value}
        onChange={(e) => handleUpdate(fullPath, parseFloat(e.target.value))}
        className="h-8 text-sm"
      />
    );
  }
  if (Array.isArray(value)) {
      return (
          <Textarea
              id={id}
              value={value.join(', ')}
              onChange={(e) => handleUpdate(fullPath, e.target.value.split(',').map(item => item.trim()))}
              className="h-8 text-sm"
          />
      )
  }
  if (typeof value === 'object' && value !== null) {
    return (
      <div className="pl-4 border-l-2 border-border space-y-2">
        {Object.entries(value).map(([subKey, subValue]) => (
          <div key={subKey}>
            <Label htmlFor={`${id}-${subKey}`} className="text-xs font-semibold">{subKey}</Label>
            {renderValueInput(subKey, subValue, section, fullPath, handleUpdate)}
          </div>
        ))}
      </div>
    )
  }
  return (
    <Input
      id={id}
      value={value as string}
      onChange={(e) => handleUpdate(fullPath, e.target.value)}
      className="h-8 text-sm"
    />
  );
};


export function Configuration() {
  const queryClient = useQueryClient()
  const { data: config, isLoading, error } = useQuery({ queryKey: ['config'], queryFn: fetchConfig })
  const [localConfig, setLocalConfig] = useState<Record<string, unknown> | null>(null)
  const [activeTab, setActiveTab] = useState('system')
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    if (config) {
      setLocalConfig(JSON.parse(JSON.stringify(config))) // Deep copy
      setHasChanges(false)
    }
  }, [config])

  const updateMutation = useMutation({ 
    mutationFn: updateConfigSection, 
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] })
      toast({ title: 'Configuration Saved', description: 'Changes saved successfully.' })
    },
    onError: (error: Error) => {
      toast({ title: 'Save Failed', description: error.message, variant: 'destructive' })
    }
  })

  // const deleteMutation = useMutation({ 
  //   mutationFn: deleteConfigKey, 
  //   onSuccess: (data, variables) => {
  //     queryClient.invalidateQueries({ queryKey: ['config'] })
  //     toast({ title: 'Key Deleted', description: `Key '${variables.key}' from section '${variables.section}' deleted.` })
  //   },
  //   onError: (error: Error) => {
  //     toast({ title: 'Delete Failed', description: error.message, variant: 'destructive' })
  //   }
  // })

  const handleUpdate = (path: string[], value: unknown) => {
    setLocalConfig((prev: Record<string, unknown> | null) => {
      const newConfig = { ...prev };
      let current = newConfig;
      for (let i = 0; i < path.length - 1; i++) {
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
      return newConfig;
    });
    setHasChanges(true);
  };
  
  // const handleDelete = (section: string, key: string) => {
  //   if (window.confirm(`Are you sure you want to delete '${key}' from the '${section}' section?`)) {
  //     deleteMutation.mutate({ section, key });
  //   }
  // };

  const handleSave = () => {
    if (!localConfig) return
    const changedSections = Object.keys(localConfig).filter(section => 
      JSON.stringify(localConfig[section]) !== JSON.stringify(config[section])
    );

    changedSections.forEach(section => {
      updateMutation.mutate({ section, data: localConfig[section] })
    });
  }

  if (isLoading) return <div className="p-6">Loading configuration...</div>
  if (error) return <div className="p-6 text-destructive">Error loading configuration: {error.message}</div>
  if (!localConfig) return <div className="p-6">No configuration loaded.</div>

  const TABS = [
      { id: 'system', label: 'System', icon: Settings },
      { id: 'broker', label: 'Broker', icon: Server },
      { id: 'trading', label: 'Trading', icon: TrendingUp },
      { id: 'model_training', label: 'ML Model', icon: Brain },
      { id: 'performance', label: 'Performance', icon: Database },
      { id: 'advanced', label: 'Advanced', icon: Shield },
  ]

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Configuration</h1>
          <p className="text-muted-foreground mt-1">
            Configure all system parameters and settings
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {hasChanges && (
            <Badge variant="destructive" className="animate-pulse">
              <AlertCircle className="h-3 w-3 mr-1" />
              Unsaved Changes
            </Badge>
          )}
          <Button variant="outline" onClick={() => { /* Import logic */ }}>
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
          <Button variant="outline" onClick={() => { /* Export logic */ }}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button onClick={handleSave} disabled={!hasChanges || updateMutation.isPending}>
            <Save className="h-4 w-4 mr-2" />
            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          {TABS.map(tab => (
              <TabsTrigger key={tab.id} value={tab.id}>{tab.label}</TabsTrigger>
          ))}
        </TabsList>

        {TABS.map(tab => (
          <TabsContent key={tab.id} value={tab.id} className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <tab.icon className="h-5 w-5" />
                  <span>{tab.label} Settings</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {localConfig[tab.id] ? Object.entries(localConfig[tab.id]).map(([key, value]) => (
                  <div key={key} className="grid grid-cols-3 gap-4 items-start">
                    <Label htmlFor={`${tab.id}-${key}`} className="pt-1 text-sm font-semibold">{key}</Label>
                    <div className="col-span-2">
                      {renderValueInput(key, value, tab.id, [tab.id], handleUpdate)}
                    </div>
                  </div>
                )) : <p>No configuration for this section.</p>}
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}