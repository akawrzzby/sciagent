/**
 * The model switcher.
 *
 * The footer of the sidebar shows which model is answering; clicking it opens
 * this panel, which lists the endpoints the deployment knows about and lets
 * the operator add another one.  An added endpoint is persisted on the server,
 * so it survives a restart, and the key never comes back to the browser.
 */
import { useEffect, useState } from 'react'
import { api, type ModelProfile } from '../lib/api'
import { Modal, Toast } from './ui'

interface Props {
  open: boolean
  onClose: () => void
  onChanged?: () => void
}

const EMPTY = { label: '', base_url: '', model: '', api_key: '' }

export function ModelDialog({ open, onClose, onChanged }: Props) {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [form, setForm] = useState({ ...EMPTY })
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [probe, setProbe] = useState<string>('')

  const reload = () => {
    api
      .models()
      .then((data) => {
        setProfiles(data.profiles)
        setActive(data.active)
      })
      .catch((err) => setToast(`读取模型列表失败：${(err as Error).message}`))
  }

  useEffect(() => {
    if (open) reload()
  }, [open])

  const use = async (profile: ModelProfile) => {
    setBusy(true)
    try {
      await api.activateModel(profile.id)
      setActive(profile.id)
      onChanged?.()
      setToast(`已切换到 ${profile.label}`)
    } catch (err) {
      setToast(`切换失败：${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (profile: ModelProfile) => {
    setBusy(true)
    try {
      await api.deleteModel(profile.id)
      reload()
      onChanged?.()
    } catch (err) {
      setToast(`删除失败：${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const test = async () => {
    setProbe('测试中…')
    try {
      const result = await api.probeModel({
        base_url: form.base_url,
        model: form.model,
        api_key: form.api_key,
      })
      if (result.ok) {
        const names = result.models ?? []
        setProbe(names.length ? `连接正常，可用模型：${names.slice(0, 6).join('、')}` : '连接正常')
        if (!form.model && names.length === 1) setForm({ ...form, model: names[0] })
      } else {
        setProbe(`连接失败：${result.error ?? '未知错误'}`)
      }
    } catch (err) {
      setProbe(`连接失败：${(err as Error).message}`)
    }
  }

  const submit = async () => {
    if (!form.base_url.trim() || !form.model.trim()) {
      setToast('接口地址和模型名不能为空')
      return
    }
    setBusy(true)
    try {
      const created = await api.addModel({
        label: form.label.trim() || form.model.trim(),
        base_url: form.base_url.trim(),
        model: form.model.trim(),
        api_key: form.api_key.trim(),
      })
      await api.activateModel(created.id)
      setForm({ ...EMPTY })
      setProbe('')
      reload()
      onChanged?.()
      setToast('已添加并启用')
    } catch (err) {
      setToast(`添加失败：${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="模型设置" wide>
      <div className="space-y-5">
        {toast && <Toast message={toast} kind="info" />}

        <section>
          <p className="mb-2 font-sans text-xs text-ink-500">
            当前使用哪一个接口来回答问题。切换后立即生效，无需重启。
          </p>
          <div className="space-y-1.5">
            {profiles.map((profile) => (
              <div
                key={profile.id}
                className={`flex items-center gap-3 rounded-lg border px-3 py-2 ${
                  active === profile.id
                    ? 'border-accent-300 bg-accent-50/60 dark:border-accent-700 dark:bg-accent-950/30'
                    : 'border-ink-200 dark:border-ink-800'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-sans text-sm font-medium text-ink-800 dark:text-parchment-100">
                      {profile.label}
                    </span>
                    {active === profile.id && (
                      <span className="chip !px-1.5 !py-0 text-[10px]">使用中</span>
                    )}
                  </div>
                  <div className="truncate font-mono text-[11px] text-ink-400">
                    {profile.model} @ {profile.base_url}
                    {profile.has_key && `  key${profile.key_hint}`}
                  </div>
                </div>
                {active === profile.id ? null : (
                  <button className="btn-secondary !py-1 !text-[11px]" disabled={busy} onClick={() => void use(profile)}>
                    使用
                  </button>
                )}
                {profiles.length > 1 && (
                  <button
                    className="btn-ghost !px-2 !py-1 text-[11px] text-red-600"
                    disabled={busy}
                    onClick={() => void remove(profile)}
                  >
                    删除
                  </button>
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-ink-200 p-3 dark:border-ink-800">
          <h3 className="mb-2 font-sans text-sm font-semibold text-ink-800 dark:text-parchment-100">
            添加一个接口
          </h3>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <label className="text-xs text-ink-500">
              显示名称
              <input
                className="input mt-1 w-full"
                placeholder="例如 本地 Qwen、DeepSeek"
                value={form.label}
                onChange={(e) => setForm({ ...form, label: e.target.value })}
              />
            </label>
            <label className="text-xs text-ink-500">
              模型名
              <input
                className="input mt-1 w-full"
                placeholder="例如 deepseek-chat"
                value={form.model}
                onChange={(e) => setForm({ ...form, model: e.target.value })}
              />
            </label>
            <label className="text-xs text-ink-500 sm:col-span-2">
              接口地址（OpenAI 兼容，通常以 /v1 结尾）
              <input
                className="input mt-1 w-full font-mono"
                placeholder="https://api.example.com/v1"
                value={form.base_url}
                onChange={(e) => setForm({ ...form, base_url: e.target.value })}
              />
            </label>
            <label className="text-xs text-ink-500 sm:col-span-2">
              API Key（本地模型可留空）
              <input
                className="input mt-1 w-full font-mono"
                type="password"
                placeholder="sk-…"
                value={form.api_key}
                onChange={(e) => setForm({ ...form, api_key: e.target.value })}
              />
            </label>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <button className="btn-secondary !py-1 !text-[11px]" onClick={() => void test()}>
              测试连接
            </button>
            <button className="btn-primary !py-1 !text-[11px]" disabled={busy} onClick={() => void submit()}>
              添加并启用
            </button>
            {probe && <span className="font-sans text-[11px] text-ink-500">{probe}</span>}
          </div>
        </section>
      </div>
    </Modal>
  )
}
