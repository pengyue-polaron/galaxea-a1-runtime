const token = document.querySelector('meta[name="operator-panel-token"]').content;
const state = { catalog: null, status: null, activePanel: null };

function element(tag, attributes = {}, text = '') {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (name === 'class') node.className = value;
    else if (name === 'dataset') Object.assign(node.dataset, value);
    else if (name in node) node[name] = value;
    else node.setAttribute(name, value);
  }
  if (text) node.textContent = text;
  return node;
}

async function request(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

async function post(path, payload) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Operator-Panel-Token': token },
    body: JSON.stringify(payload),
  });
}

function toast(message) {
  const node = document.getElementById('toast');
  node.textContent = message;
  node.classList.add('show');
  window.setTimeout(() => node.classList.remove('show'), 2800);
}

function refreshCameras() {
  document.querySelectorAll('[data-camera-url]').forEach(image => {
    image.src = `${image.dataset.cameraUrl}?panel_refresh=${Date.now()}`;
  });
}

function renderProduct() {
  document.getElementById('product-brand').textContent = state.catalog.product.brand;
  document.getElementById('product-title').textContent = state.catalog.product.title;
  document.title = `${state.catalog.product.brand} ${state.catalog.product.title}`;
}

function renderCameras() {
  const section = document.getElementById('camera-section');
  section.classList.toggle('hidden', !state.catalog.cameras.length);
  const grid = document.getElementById('camera-grid');
  grid.replaceChildren(...state.catalog.cameras.map(camera => {
    const image = element('img', { alt: `${camera.label} camera stream` });
    image.dataset.cameraUrl = camera.url;
    const figure = element('figure');
    figure.append(image, element('figcaption', {}, camera.label));
    return figure;
  }));
  const controls = document.getElementById('camera-controls');
  controls.replaceChildren(...state.catalog.camera_controls.map(control => {
    const button = element('button', { type: 'button', class: 'quiet' }, control.label);
    button.dataset.startWorkflow = control.workflow;
    button.addEventListener('click', async () => {
      if (control.confirm && !window.confirm(control.confirm)) return;
      try {
        await startWorkflow(control.workflow, control.values);
        window.setTimeout(refreshCameras, 1000);
      } catch (error) { toast(error.message); }
    });
    return button;
  }));
  refreshCameras();
}

function fieldControl(field) {
  if (field.type === 'checkbox') {
    const input = element('input', { type: 'checkbox', name: field.name });
    input.checked = Boolean(field.default);
    const label = element('label', { class: 'check' });
    label.append(input, document.createTextNode(field.label));
    return label;
  }

  let control;
  if (field.type === 'select') {
    control = element('select', { name: field.name, required: field.required !== false });
    control._field = field;
  } else {
    control = element('input', {
      type: 'text',
      name: field.name,
      required: field.required !== false,
      placeholder: field.placeholder || '',
      value: field.default || '',
    });
  }
  const label = element('label', {}, field.label);
  label.append(control);
  return label;
}

function updateSelects(form, workflow) {
  for (const field of workflow.fields.filter(item => item.type === 'select')) {
    const select = form.elements.namedItem(field.name);
    const previous = select.value || field.default || '';
    const dependency = field.depends_on
      ? form.elements.namedItem(field.depends_on).value
      : null;
    const options = field.options.filter(item => (
      !field.depends_on || item.depends_value === dependency
    ));
    select.replaceChildren(...options.map(item => element('option', {
      value: item.value,
      selected: item.value === previous,
    }, item.label)));
    if (![...select.options].some(item => item.value === select.value) && select.options.length) {
      select.selectedIndex = 0;
    }
  }
}

function workflowValues(form, workflow) {
  return Object.fromEntries(workflow.fields.map(field => {
    const control = form.elements.namedItem(field.name);
    return [field.name, field.type === 'checkbox' ? control.checked : control.value];
  }));
}

function renderWorkflows() {
  const tabs = document.getElementById('workflow-tabs');
  const panels = document.getElementById('workflow-panels');
  tabs.replaceChildren();
  panels.replaceChildren();

  for (const workflow of state.catalog.workflows) {
    const tab = element('button', { type: 'button', class: 'tab' }, workflow.label);
    tab.dataset.panel = workflow.id;
    tabs.append(tab);

    const form = element('form', { class: 'workflow-panel', id: `workflow-${workflow.id}` });
    form.append(
      element('p', { class: 'eyebrow' }, workflow.eyebrow),
      element('h2', {}, workflow.title),
      ...workflow.fields.map(fieldControl),
    );
    const submit = element('button', { type: 'submit', class: 'primary' }, workflow.submit_label);
    submit.dataset.startWorkflow = workflow.id;
    form.append(submit);
    form.querySelectorAll('select').forEach(select => {
      select.addEventListener('change', () => updateSelects(form, workflow));
    });
    updateSelects(form, workflow);
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (workflow.confirm && !window.confirm(workflow.confirm)) return;
      try { await startWorkflow(workflow.id, workflowValues(form, workflow)); }
      catch (error) { toast(error.message); }
    });
    panels.append(form);
  }

  if (state.catalog.config_types.length) {
    const tab = element('button', { type: 'button', class: 'tab' }, 'Configurations');
    tab.dataset.panel = 'configuration';
    tabs.append(tab);
  }
  tabs.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => activatePanel(tab.dataset.panel));
  });
  const defaultPanel = state.catalog.workflows[0]?.id
    || (state.catalog.config_types.length ? 'configuration' : null);
  if (defaultPanel) activatePanel(state.activePanel || defaultPanel);
}

function activatePanel(panelId) {
  state.activePanel = panelId;
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.panel === panelId);
  });
  document.querySelectorAll('.workflow-panel').forEach(panel => {
    const expected = panelId === 'configuration' ? 'configuration-panel' : `workflow-${panelId}`;
    panel.classList.toggle('active', panel.id === expected);
  });
}

function renderConfigurationEditor() {
  const kind = document.getElementById('config-kind');
  kind.replaceChildren(...state.catalog.config_types.map(item => (
    element('option', { value: item.id }, item.label)
  )));
  updateConfigTemplates();
}

function updateConfigTemplates() {
  const kindId = document.getElementById('config-kind').value;
  const definition = state.catalog.config_types.find(item => item.id === kindId);
  const template = document.getElementById('config-template');
  template.replaceChildren(...(definition ? definition.templates : []).map(item => (
    element('option', { value: item.value }, item.label)
  )));
}

async function startWorkflow(workflow, values) {
  state.status = await post('/api/start', { workflow, values });
  renderStatus();
}

function renderStatus() {
  const status = state.status || {
    active: false, name: '', logs: [], input_actions: [],
  };
  document.getElementById('status-dot').classList.toggle('active', status.active);
  document.getElementById('status-label').textContent = status.active ? 'Running' : 'Idle';
  document.getElementById('session-name').textContent = status.name || 'No active workflow';
  const logs = document.getElementById('logs');
  const atBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 40;
  logs.textContent = status.logs.length ? status.logs.join('\n') : 'Panel ready.';
  if (atBottom) logs.scrollTop = logs.scrollHeight;

  const actions = document.getElementById('session-actions');
  actions.replaceChildren(...status.input_actions.map(action => {
    const button = element('button', { type: 'button' }, action.label);
    button.addEventListener('click', async () => {
      actions.querySelectorAll('button').forEach(item => { item.disabled = true; });
      try {
        state.status = await post('/api/input', { action: action.id });
        renderStatus();
      } catch (error) { toast(error.message); }
    });
    return button;
  }));
  document.getElementById('stop-workflow').disabled = !status.active;
  document.querySelectorAll('[data-start-workflow]').forEach(button => {
    button.disabled = status.active;
  });
  document.getElementById('create-config').disabled = status.active;
}

async function pollStatus() {
  try {
    state.status = await request('/api/status');
    renderStatus();
  } catch (_error) {
    document.getElementById('status-label').textContent = 'Disconnected';
  }
}

async function loadCatalog() {
  state.catalog = await request('/api/catalog');
  renderProduct();
  renderCameras();
  renderWorkflows();
  renderConfigurationEditor();
  renderStatus();
}

document.getElementById('config-kind').addEventListener('change', updateConfigTemplates);

document.getElementById('load-template').addEventListener('click', async () => {
  try {
    const source = document.getElementById('config-template').value;
    const payload = await post('/api/config/template', {
      kind: document.getElementById('config-kind').value,
      source,
    });
    document.getElementById('config-content').value = payload.content;
    document.getElementById('config-filename').value = `${source.split('/').pop().replace(/\.toml$/, '')}_copy.toml`;
    toast('Template loaded.');
  } catch (error) { toast(error.message); }
});

document.getElementById('validate-config').addEventListener('click', async () => {
  try {
    const result = await post('/api/config/validate', configPayload());
    toast(`Valid: ${result.path}`);
  } catch (error) { toast(error.message); }
});

document.getElementById('configuration-panel').addEventListener('submit', async event => {
  event.preventDefault();
  if (!window.confirm('Create this new validated repository configuration?')) return;
  try {
    const result = await post('/api/config/create', configPayload());
    state.catalog = result.catalog;
    document.getElementById('config-filename').value = '';
    document.getElementById('config-content').value = '';
    renderProduct();
    renderCameras();
    renderWorkflows();
    renderConfigurationEditor();
    toast(`Created: ${result.created}`);
  } catch (error) { toast(error.message); }
});

function configPayload() {
  return {
    kind: document.getElementById('config-kind').value,
    filename: document.getElementById('config-filename').value,
    content: document.getElementById('config-content').value,
  };
}

document.getElementById('stop-workflow').addEventListener('click', async () => {
  if (!window.confirm('Interrupt the active workflow and let it run cleanup?')) return;
  try {
    state.status = await post('/api/stop', {});
    renderStatus();
  } catch (error) { toast(error.message); }
});

loadCatalog().catch(error => toast(error.message));
pollStatus();
window.setInterval(pollStatus, 800);
