/**
 * Recosys Intelligence Hub - Unified 3D Latent Universe & Results Explorer
 * Powered by Three.js & TWEEN.js
 */

(function () {
  // Load unified dataset
  const data = window.RECOSYS_DATA || {};
  const items = data.items || [];
  const customers = data.customers || [];
  const inventory = data.inventory || [];
  const seasonalMonthly = data.seasonal_monthly || [];
  const seasonalPeriod = data.seasonal_period || [];

  // All 3D entities: items first, then customers
  const allEntities = [...items, ...customers];

  // State
  let currentProjection = 'tsne'; // 'tsne' or 'pca'
  let entityFilterMode = 'all'; // 'all', 'items', 'customers'
  let autoRotate = true;
  let selectedEntity = null;
  let hoveredEntity = null;
  let activeClusterFilter = null;
  let search3DQuery = '';
  let activeTab = 'tab-3d';

  // Three.js Objects
  let scene, camera, renderer, controls;
  let particleSystem, particleGeometry, particleMaterial;
  let connectionLinesMesh;
  let starfield;
  let raycaster, mouse;
  let currentPositions, targetPositions;
  let colorsArray, sizesArray;

  const DEFAULT_CAM_POS = { x: 0, y: 35, z: 125 };

  // Fast lookups
  const itemMap = new Map();
  items.forEach(it => itemMap.set(String(it.id), it));

  const custMap = new Map();
  customers.forEach(c => custMap.set(String(c.id), c));

  function init() {
    setupThreeScene();
    createStarfield();
    createParticleSystem();
    createConnectionLines();
    setup3DControls();
    setupTabNavigation();
    setupResultsTabs();
    setupEventListeners();
    animate();
  }

  function setupThreeScene() {
    const container = document.getElementById('canvas-container');
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0a0c14, 0.0032);

    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.5, 2500);
    camera.position.set(DEFAULT_CAM_POS.x, DEFAULT_CAM_POS.y, DEFAULT_CAM_POS.z);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x0a0c14, 1);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 1.2;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.7;
    controls.maxDistance = 700;
    controls.minDistance = 5;

    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 2.0;
    mouse = new THREE.Vector2(-9999, -9999);

    // Subtle 3D Grid
    const grid = new THREE.GridHelper(240, 24, 0x1e293b, 0x0f172a);
    grid.position.y = -50;
    scene.add(grid);
  }

  function createStarfield() {
    const starCount = 1400;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount * 3; i += 3) {
      starPos[i] = (Math.random() - 0.5) * 1400;
      starPos[i + 1] = (Math.random() - 0.5) * 1400;
      starPos[i + 2] = (Math.random() - 0.5) * 1400;
    }

    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0x475569,
      size: 1.1,
      transparent: true,
      opacity: 0.4,
    });
    starfield = new THREE.Points(starGeo, starMat);
    scene.add(starfield);
  }

  function getGlowSpriteTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.25, 'rgba(255, 255, 255, 0.9)');
    grad.addColorStop(0.65, 'rgba(255, 255, 255, 0.25)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    return texture;
  }

  function createParticleSystem() {
    const count = allEntities.length;
    particleGeometry = new THREE.BufferGeometry();

    currentPositions = new Float32Array(count * 3);
    targetPositions = new Float32Array(count * 3);
    colorsArray = new Float32Array(count * 3);
    sizesArray = new Float32Array(count);

    const projKey = currentProjection === 'tsne' ? 'tsne' : 'pca';

    for (let i = 0; i < count; i++) {
      const entity = allEntities[i];
      const pos = entity[projKey];

      currentPositions[i * 3] = pos.x;
      currentPositions[i * 3 + 1] = pos.y;
      currentPositions[i * 3 + 2] = pos.z;

      targetPositions[i * 3] = pos.x;
      targetPositions[i * 3 + 1] = pos.y;
      targetPositions[i * 3 + 2] = pos.z;

      const c = new THREE.Color(entity.color);
      colorsArray[i * 3] = c.r;
      colorsArray[i * 3 + 1] = c.g;
      colorsArray[i * 3 + 2] = c.b;

      // Customer vectors are slightly larger and brighter
      sizesArray[i] = entity.type === 'customer' ? 5.5 : 3.5;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(currentPositions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
    particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizesArray, 1));

    particleMaterial = new THREE.PointsMaterial({
      size: 4.0,
      vertexColors: true,
      map: getGlowSpriteTexture(),
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    particleSystem = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particleSystem);
  }

  function createConnectionLines() {
    const lineGeo = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
    });
    connectionLinesMesh = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(connectionLinesMesh);
  }

  function updateConnectionLines(entity) {
    if (!entity) {
      connectionLinesMesh.geometry.dispose();
      connectionLinesMesh.geometry = new THREE.BufferGeometry();
      return;
    }

    const posAttr = particleGeometry.attributes.position;
    const originIdx = entity.index;
    const ox = posAttr.getX(originIdx);
    const oy = posAttr.getY(originIdx);
    const oz = posAttr.getZ(originIdx);

    const linePoints = [];
    const lineColors = [];

    if (entity.type === 'item') {
      // Connect item to its top 5 nearest neighbor items (cyan lines)
      const cCyan = new THREE.Color(0x00f0ff);
      (entity.neighbors || []).forEach(n => {
        const neighborIdx = n.index;
        if (neighborIdx !== undefined && neighborIdx < allEntities.length) {
          linePoints.push(ox, oy, oz);
          linePoints.push(
            posAttr.getX(neighborIdx),
            posAttr.getY(neighborIdx),
            posAttr.getZ(neighborIdx)
          );
          lineColors.push(cCyan.r, cCyan.g, cCyan.b);
          lineColors.push(cCyan.r, cCyan.g, cCyan.b);
        }
      });
    } else if (entity.type === 'customer') {
      // 1. Connect customer to Lookalike Peers (Gold lines)
      const cGold = new THREE.Color(0xffd700);
      (entity.lookalikes || []).forEach(peer => {
        const targetCust = custMap.get(String(peer.customer_id));
        if (targetCust) {
          const peerIdx = targetCust.index;
          linePoints.push(ox, oy, oz);
          linePoints.push(
            posAttr.getX(peerIdx),
            posAttr.getY(peerIdx),
            posAttr.getZ(peerIdx)
          );
          lineColors.push(cGold.r, cGold.g, cGold.b);
          lineColors.push(cGold.r, cGold.g, cGold.b);
        }
      });

      // 2. Connect customer to Recommended Items (Cyan lines)
      const cCyan = new THREE.Color(0x00f0ff);
      const allRecs = [...(entity.personalized_recs || []).slice(0, 5), ...(entity.peer_recs || []).slice(0, 5)];
      allRecs.forEach(r => {
        const targetItem = itemMap.get(String(r.item_id));
        if (targetItem) {
          const itIdx = targetItem.index;
          linePoints.push(ox, oy, oz);
          linePoints.push(
            posAttr.getX(itIdx),
            posAttr.getY(itIdx),
            posAttr.getZ(itIdx)
          );
          lineColors.push(cCyan.r, cCyan.g, cCyan.b);
          lineColors.push(cCyan.r, cCyan.g, cCyan.b);
        }
      });
    }

    connectionLinesMesh.geometry.dispose();
    const newGeo = new THREE.BufferGeometry();
    newGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePoints, 3));
    newGeo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
    connectionLinesMesh.geometry = newGeo;
  }

  function switchProjection(mode) {
    if (mode === currentProjection) return;
    currentProjection = mode;

    document.getElementById('btn-tsne').classList.toggle('active', mode === 'tsne');
    document.getElementById('btn-pca').classList.toggle('active', mode === 'pca');
    document.getElementById('dock-proj-name').textContent = mode.toUpperCase() + ' 3D';

    const count = allEntities.length;
    const fromPositions = new Float32Array(currentPositions);
    const destPositions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const p = allEntities[i][mode];
      destPositions[i * 3] = p.x;
      destPositions[i * 3 + 1] = p.y;
      destPositions[i * 3 + 2] = p.z;
    }

    const animObj = { progress: 0 };
    new TWEEN.Tween(animObj)
      .to({ progress: 1 }, 1100)
      .easing(TWEEN.Easing.Cubic.InOut)
      .onUpdate(() => {
        const t = animObj.progress;
        const posAttr = particleGeometry.attributes.position;
        for (let i = 0; i < count * 3; i++) {
          currentPositions[i] = fromPositions[i] + (destPositions[i] - fromPositions[i]) * t;
        }
        posAttr.needsUpdate = true;
        if (selectedEntity) {
          updateConnectionLines(selectedEntity);
        }
      })
      .onComplete(() => {
        for (let i = 0; i < count * 3; i++) {
          currentPositions[i] = destPositions[i];
        }
        particleGeometry.attributes.position.needsUpdate = true;
      })
      .start();
  }

  function flyCameraTo(targetPos, distance = 35) {
    controls.autoRotate = false;
    document.getElementById('btn-autorotate').classList.remove('active');

    const startPos = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
    const startTarget = { x: controls.target.x, y: controls.target.y, z: controls.target.z };

    const offset = new THREE.Vector3()
      .subVectors(camera.position, controls.target)
      .normalize()
      .multiplyScalar(distance);

    const endCamPos = {
      x: targetPos.x + offset.x,
      y: targetPos.y + offset.y + 4,
      z: targetPos.z + offset.z,
    };

    new TWEEN.Tween(startPos)
      .to(endCamPos, 900)
      .easing(TWEEN.Easing.Cubic.Out)
      .onUpdate(() => {
        camera.position.set(startPos.x, startPos.y, startPos.z);
      })
      .start();

    new TWEEN.Tween(startTarget)
      .to(targetPos, 900)
      .easing(TWEEN.Easing.Cubic.Out)
      .onUpdate(() => {
        controls.target.set(startTarget.x, startTarget.y, startTarget.z);
      })
      .start();
  }

  function selectEntity(entity, animateCam = true) {
    selectedEntity = entity;
    if (!entity) {
      updateConnectionLines(null);
      document.getElementById('detail-card').classList.remove('open');
      return;
    }

    const pos = entity[currentProjection];
    if (animateCam) {
      flyCameraTo({ x: pos.x, y: pos.y, z: pos.z }, entity.type === 'customer' ? 40 : 32);
    }
    updateConnectionLines(entity);
    renderDetailCard(entity);
  }

  function renderDetailCard(entity) {
    const card = document.getElementById('detail-card');
    const badge = document.getElementById('card-badge');
    const grid = document.getElementById('card-grid');
    const connTitle = document.getElementById('card-connection-title');
    const connList = document.getElementById('card-connections-list');

    card.querySelectorAll('.ai-dynamic-card').forEach(el => el.remove());
    connList.innerHTML = '';

    if (entity.type === 'item') {
      badge.textContent = entity.cluster_name ? `COLLECTION • ${entity.cluster_name}` : `ITEM • COHORT #${entity.cluster_id + 1}`;
      badge.style.background = `${entity.color}22`;
      badge.style.color = entity.color;
      badge.style.border = `1px solid ${entity.color}66`;

      document.getElementById('card-title').textContent = entity.name || `Item ${entity.code}`;

      grid.innerHTML = `
        <div class="grid-cell">
          <span class="cell-label">Item Code</span>
          <span class="cell-value cyan">${entity.code}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Category</span>
          <span class="cell-value">${entity.category}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Retail Price</span>
          <span class="cell-value">$${entity.price.toFixed(2)}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Current Stock</span>
          <span class="cell-value ${entity.stock <= 5 ? 'red' : 'green'}">${entity.stock} pcs</span>
        </div>
      `;

      if (entity.sales_pitch) {
        const pitchCard = document.createElement('div');
        pitchCard.className = 'ai-dynamic-card';
        pitchCard.style.cssText = 'margin: 12px 0 8px 0; padding: 10px 12px; background: rgba(0, 240, 255, 0.08); border-left: 3px solid var(--accent-cyan); border-radius: 4px;';
        pitchCard.innerHTML = `
          <div style="font-size: 11px; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; margin-bottom: 3px; display: flex; align-items: center; gap: 4px;">
            <span>✨ AI Sales Pitch</span>
          </div>
          <div style="font-size: 12px; line-height: 1.45; color: #f1f5f9;">${entity.sales_pitch}</div>
          ${entity.merchandising_notes ? `<div style="font-size: 11px; color: #94a3b8; margin-top: 5px; font-style: italic;">Merchandising: ${entity.merchandising_notes}</div>` : ''}
        `;
        grid.parentNode.insertBefore(pitchCard, grid.nextSibling);
      }

      connTitle.textContent = 'Top Nearest Neighbor Items (Cosine Sim)';
      (entity.neighbors || []).forEach(n => {
        const row = document.createElement('div');
        row.className = 'connected-item';
        row.innerHTML = `
          <div class="item-left">
            <span class="item-primary">${n.name || 'Item ' + n.code}</span>
            <span class="item-secondary">${n.code} • $${n.price.toFixed(2)}</span>
          </div>
          <span class="score-badge">${(n.similarity * 100).toFixed(1)}%</span>
        `;
        row.addEventListener('click', () => {
          const target = allEntities[n.index];
          if (target) selectEntity(target, true);
        });
        connList.appendChild(row);
      });

    } else if (entity.type === 'customer') {
      badge.textContent = `CUSTOMER ACCOUNT #${entity.account_number || entity.id}`;
      badge.style.background = 'rgba(255, 215, 0, 0.18)';
      badge.style.color = '#ffd700';
      badge.style.border = '1px solid rgba(255, 215, 0, 0.5)';

      document.getElementById('card-title').textContent = entity.name || `Customer #${entity.id}`;

      grid.innerHTML = `
        <div class="grid-cell">
          <span class="cell-label">Account No.</span>
          <span class="cell-value gold">${entity.account_number || 'N/A'}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Account ID</span>
          <span class="cell-value">${entity.id}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Outstanding Balance</span>
          <span class="cell-value">$${entity.balance.toFixed(2)}</span>
        </div>
        <div class="grid-cell">
          <span class="cell-label">Lookalike Peers</span>
          <span class="cell-value">${(entity.lookalikes || []).length} peers</span>
        </div>
      `;

      if (entity.buyer_persona) {
        const personaCard = document.createElement('div');
        personaCard.className = 'ai-dynamic-card';
        personaCard.style.cssText = 'margin: 12px 0 8px 0; padding: 10px 12px; background: rgba(255, 215, 0, 0.08); border-left: 3px solid var(--accent-gold); border-radius: 4px;';
        personaCard.innerHTML = `
          <div style="font-size: 11px; font-weight: 700; color: var(--accent-gold); text-transform: uppercase; margin-bottom: 3px;">
            <span>👤 Buyer Persona (Gemini AI)</span>
          </div>
          <div style="font-size: 12px; line-height: 1.45; color: #f1f5f9;">${entity.buyer_persona}</div>
          ${entity.sales_notes ? `<div style="font-size: 11px; color: #cbd5e1; margin-top: 5px;">💡 <span style="font-weight:600;">Sales Rep Guidance:</span> ${entity.sales_notes}</div>` : ''}
        `;
        grid.parentNode.insertBefore(personaCard, grid.nextSibling);
      }

      connTitle.textContent = 'Lookalike Peers & Top Recommended Items';

      // 1. Add lookalikes
      (entity.lookalikes || []).slice(0, 3).forEach(peer => {
        const row = document.createElement('div');
        row.className = 'connected-item gold-hover';
        row.innerHTML = `
          <div class="item-left">
            <span class="item-primary" style="color:var(--accent-gold);">👤 ${peer.name}</span>
            <span class="item-secondary">Account #${peer.account_number} • Peer Match</span>
          </div>
          <span class="score-badge" style="color:var(--accent-gold);">${(peer.similarity * 100).toFixed(1)}%</span>
        `;
        row.addEventListener('click', () => {
          const target = custMap.get(String(peer.customer_id));
          if (target) selectEntity(target, true);
        });
        connList.appendChild(row);
      });

      // 2. Add top recommended items
      const recs = (entity.personalized_recs || []).slice(0, 3);
      recs.forEach(r => {
        const row = document.createElement('div');
        row.className = 'connected-item';
        row.innerHTML = `
          <div class="item-left">
            <span class="item-primary">✨ ${r.name || 'Item ' + r.code}</span>
            <span class="item-secondary">${r.code} • $${r.price.toFixed(2)} • Stock: ${r.stock}</span>
          </div>
          <span class="score-badge">${(r.score * 100).toFixed(1)}%</span>
        `;
        row.addEventListener('click', () => {
          const target = itemMap.get(String(r.item_id));
          if (target) selectEntity(target, true);
        });
        connList.appendChild(row);
      });
    }

    card.classList.add('open');
  }

  function apply3DFilters() {
    const query = search3DQuery.trim().toLowerCase();

    for (let i = 0; i < allEntities.length; i++) {
      const entity = allEntities[i];

      // Entity type filter
      let matchesType = true;
      if (entityFilterMode === 'items' && entity.type !== 'item') matchesType = false;
      if (entityFilterMode === 'customers' && entity.type !== 'customer') matchesType = false;

      // Cluster filter (for items)
      let matchesCluster = true;
      if (entity.type === 'item' && activeClusterFilter !== null) {
        matchesCluster = entity.cluster_id === activeClusterFilter;
      }

      // Search filter
      let matchesSearch = true;
      if (query) {
        if (entity.type === 'item') {
          matchesSearch =
            (entity.name && entity.name.toLowerCase().includes(query)) ||
            (entity.code && entity.code.toLowerCase().includes(query)) ||
            (entity.category && entity.category.toLowerCase().includes(query));
        } else if (entity.type === 'customer') {
          matchesSearch =
            (entity.name && entity.name.toLowerCase().includes(query)) ||
            (entity.account_number && entity.account_number.toLowerCase().includes(query)) ||
            entity.id.includes(query);
        }
      }

      const isVisible = matchesType && matchesCluster && matchesSearch;

      const c = new THREE.Color(entity.color);
      if (isVisible) {
        colorsArray[i * 3] = c.r;
        colorsArray[i * 3 + 1] = c.g;
        colorsArray[i * 3 + 2] = c.b;
      } else {
        colorsArray[i * 3] = c.r * 0.08;
        colorsArray[i * 3 + 1] = c.g * 0.08;
        colorsArray[i * 3 + 2] = c.b * 0.08;
      }
    }

    particleGeometry.attributes.color.needsUpdate = true;
  }

  function setup3DControls() {
    // Entity switcher
    const btnAll = document.getElementById('btn-show-all');
    const btnItems = document.getElementById('btn-show-items');
    const btnCusts = document.getElementById('btn-show-custs');

    if (btnAll) btnAll.textContent = `All (${allEntities.length.toLocaleString()})`;
    if (btnItems) btnItems.textContent = `Items (${items.length.toLocaleString()})`;
    if (btnCusts) btnCusts.textContent = `Customers (${customers.length.toLocaleString()})`;

    const legendItems = document.getElementById('legend-items-count');
    if (legendItems) legendItems.textContent = `Catalog Items (${items.length.toLocaleString()})`;
    const legendCusts = document.getElementById('legend-custs-count');
    if (legendCusts) legendCusts.textContent = `Customer Vectors (${customers.length.toLocaleString()})`;
    const dockItems = document.getElementById('dock-total-items');
    if (dockItems) dockItems.textContent = items.length.toLocaleString();
    const dockCusts = document.getElementById('dock-total-custs');
    if (dockCusts) dockCusts.textContent = customers.length.toLocaleString();

    function setEntityFilter(mode) {
      entityFilterMode = mode;
      btnAll.classList.toggle('active', mode === 'all');
      btnItems.classList.toggle('active', mode === 'items');
      btnCusts.classList.toggle('active', mode === 'customers');
      apply3DFilters();
    }

    btnAll.addEventListener('click', () => setEntityFilter('all'));
    btnItems.addEventListener('click', () => setEntityFilter('items'));
    btnCusts.addEventListener('click', () => setEntityFilter('customers'));

    // Projections
    document.getElementById('btn-tsne').addEventListener('click', () => switchProjection('tsne'));
    document.getElementById('btn-pca').addEventListener('click', () => switchProjection('pca'));

    // Auto rotate
    const rotBtn = document.getElementById('btn-autorotate');
    rotBtn.addEventListener('click', () => {
      autoRotate = !autoRotate;
      controls.autoRotate = autoRotate;
      rotBtn.classList.toggle('active', autoRotate);
    });

    // Reset view
    document.getElementById('btn-reset-view').addEventListener('click', () => {
      flyCameraTo({ x: 0, y: 0, z: 0 }, 125);
      selectEntity(null);
    });

    // Close detail card
    document.getElementById('close-card-btn').addEventListener('click', () => {
      selectEntity(null);
    });

    // Search box
    document.getElementById('search-input-3d').addEventListener('input', (e) => {
      search3DQuery = e.target.value;
      apply3DFilters();
    });

    // Item Cohorts filter
    const clustersCount = {};
    items.forEach(it => {
      clustersCount[it.cluster_id] = (clustersCount[it.cluster_id] || 0) + 1;
    });

    const clusterList = document.getElementById('cluster-filter-list');
    const sortedClusters = Object.keys(clustersCount).sort((a, b) => a - b);

    // All collections button
    const allBtn = document.createElement('div');
    allBtn.className = 'filter-badge-item active';
    allBtn.innerHTML = `
      <div class="dot-label">
        <div class="color-dot" style="background:#fff; color:#fff;"></div>
        <span>All Collections</span>
      </div>
      <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">${items.length}</span>
    `;
    allBtn.addEventListener('click', () => {
      document.querySelectorAll('#cluster-filter-list .filter-badge-item').forEach(el => el.classList.remove('active'));
      allBtn.classList.add('active');
      activeClusterFilter = null;
      apply3DFilters();
    });
    clusterList.appendChild(allBtn);

    sortedClusters.forEach(cId => {
      const sampleItem = items.find(it => it.cluster_id === parseInt(cId));
      const color = sampleItem ? sampleItem.color : '#00f0ff';
      const clusterName = (sampleItem && sampleItem.cluster_name) ? sampleItem.cluster_name : `Collection #${parseInt(cId) + 1}`;
      const el = document.createElement('div');
      el.className = 'filter-badge-item';
      el.title = clusterName;
      el.innerHTML = `
        <div class="dot-label" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          <div class="color-dot" style="background:${color}; color:${color}; flex-shrink: 0;"></div>
          <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${clusterName}</span>
        </div>
        <span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); flex-shrink: 0;">${clustersCount[cId]}</span>
      `;
      el.addEventListener('click', () => {
        document.querySelectorAll('#cluster-filter-list .filter-badge-item').forEach(e => e.classList.remove('active'));
        el.classList.add('active');
        activeClusterFilter = parseInt(cId);
        apply3DFilters();
      });
      clusterList.appendChild(el);
    });
  }

  // Primary Tab Navigation
  function setupTabNavigation() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const viewPanels = document.querySelectorAll('.view-panel');
    const controls3D = document.getElementById('controls-3d');
    const sidebar3D = document.getElementById('sidebar-3d');
    const bottomDock = document.getElementById('bottom-dock');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        activeTab = targetTab;

        tabButtons.forEach(b => b.classList.toggle('active', b === btn));
        viewPanels.forEach(p => p.classList.toggle('active', p.id === targetTab));

        const is3D = targetTab === 'tab-3d';
        controls3D.style.display = is3D ? 'flex' : 'none';
        sidebar3D.style.display = is3D ? 'flex' : 'none';
        bottomDock.style.display = is3D ? 'flex' : 'none';

        if (is3D) {
          controls.enabled = true;
        } else {
          controls.enabled = false;
        }
      });
    });
  }

  // Switch to 3D and focus on entity
  function focusEntityIn3D(entity) {
    // Switch to 3D tab
    document.querySelector('.tab-btn[data-tab="tab-3d"]').click();
    selectEntity(entity, true);
  }

  // Setup Results Tables (Tabs 2, 3, 4, 5)
  function setupResultsTabs() {
    // ----------------------------------------------------
    // TAB 2: Customer Recs Hub
    // ----------------------------------------------------
    const custSelect = document.getElementById('cust-select');
    customers.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = `${c.name} (${c.account_number ? '#' + c.account_number : '#' + c.id})`;
      custSelect.appendChild(opt);
    });

    // Helper: switch to Tab 2 and select a customer
    function switchToCustomerRecs(cId) {
      const recsTabBtn = document.querySelector('.tab-btn[data-tab="tab-recs"]');
      if (recsTabBtn) recsTabBtn.click();
      if (custSelect) {
        custSelect.value = String(cId);
        renderCustomerRecs(cId);
      }
    }

    function renderCustomerRecs(cId) {
      const cust = custMap.get(String(cId));
      if (!cust) return;

      // Update executive customer profile header
      const nameEl = document.getElementById('cust-profile-name');
      const accEl = document.getElementById('cust-profile-acc');
      const balEl = document.getElementById('cust-profile-balance');
      const peersEl = document.getElementById('cust-profile-peers');
      if (nameEl) nameEl.textContent = cust.name;
      if (accEl) accEl.textContent = cust.account_number ? `#${cust.account_number}` : `#${cust.id}`;
      if (balEl) balEl.textContent = `$${(cust.balance || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
      if (peersEl) peersEl.textContent = `${(cust.lookalikes || []).length} peers`;

      // Update persona banner
      const personaText = document.getElementById('cust-ai-persona-text');
      const notesText = document.getElementById('cust-ai-notes-text');
      if (personaText) {
        personaText.textContent = cust.buyer_persona || "Commercial tableware wholesale client with regular replenishment cycles.";
      }
      if (notesText) {
        notesText.textContent = cust.sales_notes || "Engage on core re-order replenishment and pitch complementary banquet collections.";
      }

      // Update counts
      const directRecs = cust.personalized_recs || [];
      const peerRecs = cust.peer_recs || [];
      const countDirectEl = document.getElementById('count-direct-recs');
      const countPeerEl = document.getElementById('count-peer-recs');
      if (countDirectEl) countDirectEl.textContent = directRecs.length;
      if (countPeerEl) countPeerEl.textContent = peerRecs.length;

      // Direct recs table
      const tbodyDirect = document.getElementById('tbody-direct-recs');
      tbodyDirect.innerHTML = '';
      directRecs.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="cell-nowrap"><span style="font-family:var(--font-mono); color:var(--accent-cyan); font-weight:700;">${r.code}</span></td>
          <td><div class="cell-product-name">${r.name}</div></td>
          <td class="cell-nowrap"><span style="font-size:0.75rem; color:var(--text-muted);">${r.category}</span></td>
          <td><div class="pitch-cell">✨ ${r.pitch || 'Recommended high-affinity item.'}</div></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:600;">$${r.price.toFixed(2)}</td>
          <td class="cell-nowrap"><span class="badge-status ${r.stock <= 5 ? 'status-critical' : 'status-healthy'}">${r.stock} pcs</span></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); color:var(--accent-emerald); font-weight:700;">${(r.score * 100).toFixed(1)}%</td>
          <td class="cell-nowrap"><button class="btn-fly-3d" data-item-id="${r.item_id}">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          const it = itemMap.get(String(r.item_id));
          if (it) focusEntityIn3D(it);
        });
        tbodyDirect.appendChild(tr);
      });

      // Peer recs table
      const tbodyPeer = document.getElementById('tbody-peer-recs');
      tbodyPeer.innerHTML = '';
      peerRecs.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="cell-nowrap"><span style="font-family:var(--font-mono); color:var(--accent-gold); font-weight:700;">${r.code}</span></td>
          <td><div class="cell-product-name">${r.name}</div></td>
          <td class="cell-nowrap"><span style="font-size:0.75rem; color:var(--accent-gold); font-weight:600;">👥 ${r.via}</span></td>
          <td><div class="pitch-cell gold">✨ ${r.pitch || 'Lookalike peer purchase pattern.'}</div></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:600;">$${r.price.toFixed(2)}</td>
          <td class="cell-nowrap"><span class="badge-status ${r.stock <= 5 ? 'status-critical' : 'status-healthy'}">${r.stock} pcs</span></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); color:var(--accent-gold); font-weight:700;">${r.score.toFixed(2)}</td>
          <td class="cell-nowrap"><button class="btn-fly-3d" data-item-id="${r.item_id}">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          const it = itemMap.get(String(r.item_id));
          if (it) focusEntityIn3D(it);
        });
        tbodyPeer.appendChild(tr);
      });
    }

    // Toggle sub-modes in Customer Recs Hub
    const secDirect = document.getElementById('section-direct-recs');
    const secPeer = document.getElementById('section-peer-recs');
    const btnDirect = document.getElementById('btn-recs-direct');
    const btnPeer = document.getElementById('btn-recs-peer');
    const btnBoth = document.getElementById('btn-recs-both');
    const recsTablesContainer = document.getElementById('recs-tables-container');

    function setRecsViewMode(mode) {
      if (btnDirect) btnDirect.classList.toggle('active', mode === 'direct');
      if (btnPeer) btnPeer.classList.toggle('active', mode === 'peer');
      if (btnBoth) btnBoth.classList.toggle('active', mode === 'both');

      if (mode === 'direct') {
        if (secDirect) secDirect.style.display = 'flex';
        if (secPeer) secPeer.style.display = 'none';
        if (recsTablesContainer) {
          recsTablesContainer.style.display = 'flex';
          recsTablesContainer.style.flexDirection = 'column';
        }
      } else if (mode === 'peer') {
        if (secDirect) secDirect.style.display = 'none';
        if (secPeer) secPeer.style.display = 'flex';
        if (recsTablesContainer) {
          recsTablesContainer.style.display = 'flex';
          recsTablesContainer.style.flexDirection = 'column';
        }
      } else if (mode === 'both') {
        if (secDirect) secDirect.style.display = 'flex';
        if (secPeer) secPeer.style.display = 'flex';
        if (recsTablesContainer) {
          recsTablesContainer.style.display = 'grid';
          recsTablesContainer.style.gridTemplateColumns = '1fr 1fr';
        }
      }
    }

    if (btnDirect) btnDirect.addEventListener('click', () => setRecsViewMode('direct'));
    if (btnPeer) btnPeer.addEventListener('click', () => setRecsViewMode('peer'));
    if (btnBoth) btnBoth.addEventListener('click', () => setRecsViewMode('both'));

    custSelect.addEventListener('change', (e) => renderCustomerRecs(e.target.value));
    if (customers.length > 0) {
      renderCustomerRecs(customers[0].id);
    }

    document.getElementById('btn-locate-cust-3d').addEventListener('click', () => {
      const c = custMap.get(String(custSelect.value));
      if (c) focusEntityIn3D(c);
    });

    // ----------------------------------------------------
    // TAB 3: Customer Intelligence & Lookalike Directory
    // ----------------------------------------------------
    const tbodyPersonasDir = document.getElementById('tbody-personas-directory');
    const tbodyLookalikes = document.getElementById('tbody-lookalikes');
    const containerPersonas = document.getElementById('container-personas-directory');
    const containerPairs = document.getElementById('container-lookalikes-pairs');
    const btnTogglePersonas = document.getElementById('btn-toggle-personas');
    const btnTogglePairs = document.getElementById('btn-toggle-pairs');

    let activeLookalikeSubView = 'personas'; // 'personas' or 'pairs'

    function setLookalikeSubView(view) {
      activeLookalikeSubView = view;
      if (btnTogglePersonas) btnTogglePersonas.classList.toggle('active', view === 'personas');
      if (btnTogglePairs) btnTogglePairs.classList.toggle('active', view === 'pairs');

      if (view === 'personas') {
        if (containerPersonas) containerPersonas.style.display = 'block';
        if (containerPairs) containerPairs.style.display = 'none';
      } else {
        if (containerPersonas) containerPersonas.style.display = 'none';
        if (containerPairs) containerPairs.style.display = 'block';
      }
    }

    if (btnTogglePersonas) btnTogglePersonas.addEventListener('click', () => setLookalikeSubView('personas'));
    if (btnTogglePairs) btnTogglePairs.addEventListener('click', () => setLookalikeSubView('pairs'));

    // Populate lookalike pairs data structure
    const allLookalikeRows = [];
    customers.forEach(c => {
      (c.lookalikes || []).forEach((peer, idx) => {
        allLookalikeRows.push({
          c1_id: c.id,
          c1_name: c.name,
          c1_acc: c.account_number,
          c1_persona: c.buyer_persona,
          c2_id: peer.customer_id,
          c2_name: peer.name,
          c2_acc: peer.account_number,
          score: peer.similarity,
          rank: idx + 1,
        });
      });
    });

    function renderPersonasDirectory(query = '') {
      if (!tbodyPersonasDir) return;
      tbodyPersonasDir.innerHTML = '';
      const q = query.trim().toLowerCase();

      const filtered = customers.filter(c => {
        if (!q) return true;
        return (
          (c.name && c.name.toLowerCase().includes(q)) ||
          (c.account_number && c.account_number.toLowerCase().includes(q)) ||
          (c.id && String(c.id).toLowerCase().includes(q)) ||
          (c.buyer_persona && c.buyer_persona.toLowerCase().includes(q)) ||
          (c.sales_notes && c.sales_notes.toLowerCase().includes(q))
        );
      });

      filtered.forEach(c => {
        const topPeer = (c.lookalikes && c.lookalikes.length > 0) ? c.lookalikes[0] : null;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>
            <div style="font-weight: 700; color: #fff; cursor: pointer;" class="cust-link" title="Click to view full recommendations">${c.name}</div>
            <div style="font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono);">Client ID: ${c.id}</div>
          </td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); color:var(--accent-gold); font-weight:600;">#${c.account_number || c.id}</td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:600;">$${(c.balance || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
          <td>
            <div class="persona-cell">👤 ${c.buyer_persona || 'Wholesale commercial client.'}</div>
          </td>
          <td>
            <div class="guidance-cell">💡 ${c.sales_notes || 'Engage on primary replenishment.'}</div>
          </td>
          <td>
            ${topPeer ? `
              <div style="font-size:0.78rem; font-weight:600; color:var(--accent-cyan);">👥 ${topPeer.name}</div>
              <div style="font-size:0.72rem; color:var(--accent-emerald); font-family:var(--font-mono);">${(topPeer.similarity * 100).toFixed(1)}% match</div>
            ` : '<span style="color:var(--text-muted); font-size:0.75rem;">None</span>'}
          </td>
          <td class="cell-nowrap">
            <button class="btn-fly-3d btn-open-recs" style="background:rgba(255,215,0,0.12); color:var(--accent-gold); border-color:rgba(255,215,0,0.3); margin-bottom:4px; display:inline-flex;">🎯 Recs Hub</button>
            <button class="btn-fly-3d btn-cust-3d" style="display:inline-flex;">🌐 3D View</button>
          </td>
        `;
        tr.querySelector('.cust-link').addEventListener('click', () => switchToCustomerRecs(c.id));
        tr.querySelector('.btn-open-recs').addEventListener('click', () => switchToCustomerRecs(c.id));
        tr.querySelector('.btn-cust-3d').addEventListener('click', () => focusEntityIn3D(c));
        tbodyPersonasDir.appendChild(tr);
      });
    }

    function renderLookalikes(query = '') {
      if (!tbodyLookalikes) return;
      tbodyLookalikes.innerHTML = '';
      const q = query.trim().toLowerCase();

      const filtered = allLookalikeRows.filter(r => {
        if (!q) return true;
        return (
          r.c1_name.toLowerCase().includes(q) ||
          r.c2_name.toLowerCase().includes(q) ||
          (r.c1_acc && r.c1_acc.toLowerCase().includes(q)) ||
          (r.c2_acc && r.c2_acc.toLowerCase().includes(q)) ||
          (r.c1_persona && r.c1_persona.toLowerCase().includes(q))
        );
      });

      filtered.slice(0, 150).forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span style="font-weight:600; color:#fff;">${r.c1_name}</span></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono);">#${r.c1_acc || 'N/A'}</td>
          <td><div class="persona-cell" style="font-size:0.74rem; max-width:240px;">👤 ${r.c1_persona || 'Client'}</div></td>
          <td><span style="font-weight:600; color:var(--accent-gold);">👤 ${r.c2_name}</span></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono);">#${r.c2_acc || 'N/A'}</td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); color:var(--accent-emerald); font-weight:700;">${(r.score * 100).toFixed(1)}%</td>
          <td class="cell-nowrap"><span class="badge-status" style="background:rgba(255,255,255,0.06);">Rank #${r.rank}</span></td>
          <td class="cell-nowrap"><button class="btn-fly-3d">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          const c = custMap.get(String(r.c1_id));
          if (c) focusEntityIn3D(c);
        });
        tbodyLookalikes.appendChild(tr);
      });
    }

    renderPersonasDirectory();
    renderLookalikes();

    document.getElementById('search-lookalike').addEventListener('input', (e) => {
      const q = e.target.value;
      renderPersonasDirectory(q);
      renderLookalikes(q);
    });

    // ----------------------------------------------------
    // TAB 4: Inventory & ROP Command Center
    // ----------------------------------------------------
    const tbodyInv = document.getElementById('tbody-inventory');
    let invFilterStatus = 'ALL';
    let invSearch = '';

    function renderInventory() {
      tbodyInv.innerHTML = '';
      const filtered = inventory.filter(it => {
        const matchesStatus = invFilterStatus === 'ALL' || it.stock_status === invFilterStatus;
        let matchesSearch = true;
        if (invSearch) {
          const q = invSearch.toLowerCase();
          const itemEntity = itemMap.get(String(it.item_id));
          matchesSearch =
            (it.code && it.code.toLowerCase().includes(q)) ||
            (it.name && it.name.toLowerCase().includes(q)) ||
            (it.category_name && it.category_name.toLowerCase().includes(q)) ||
            (itemEntity && itemEntity.cluster_name && itemEntity.cluster_name.toLowerCase().includes(q));
        }
        return matchesStatus && matchesSearch;
      });

      filtered.slice(0, 200).forEach(it => {
        let badgeClass = 'status-healthy';
        if (it.stock_status === 'CRITICAL_SAFETY_BREACH') badgeClass = 'status-critical';
        else if (it.stock_status === 'REORDER_NOW') badgeClass = 'status-reorder';
        else if (it.stock_status === 'OVERSTOCKED') badgeClass = 'status-overstocked';

        const itemEntity = itemMap.get(String(it.item_id));
        const clusterName = itemEntity ? (itemEntity.cluster_name || 'General') : 'General';
        const clusterColor = itemEntity ? itemEntity.color : '#00f0ff';

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="cell-nowrap"><span style="font-family:var(--font-mono); color:var(--accent-cyan); font-weight:700;">${it.code}</span></td>
          <td><div class="cell-product-name">${it.name}</div></td>
          <td class="cell-nowrap"><span class="badge-cluster" style="background:${clusterColor}22; color:${clusterColor}; border:1px solid ${clusterColor}55;">${clusterName}</span></td>
          <td class="cell-nowrap"><span style="font-size:0.75rem; color:var(--text-muted);">${it.category_name}</span></td>
          <td><div class="pitch-cell">✨ ${it.sales_pitch || 'Core catalog collection.'}</div></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:700;">${it.current_stock}</td>
          <td class="cell-nowrap" style="font-family:var(--font-mono);">${it.avg_daily_demand.toFixed(2)}/d</td>
          <td class="cell-nowrap" style="font-family:var(--font-mono);">${it.safety_stock}</td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:700; color:var(--accent-gold);">${it.reorder_point}</td>
          <td class="cell-nowrap"><span class="badge-status ${badgeClass}">${it.stock_status.replace(/_/g, ' ')}</span></td>
          <td class="cell-nowrap" style="font-family:var(--font-mono); font-weight:600;">$${it.retail_price.toFixed(2)}</td>
          <td class="cell-nowrap"><span style="font-family:var(--font-mono); font-size:0.74rem; color:var(--text-secondary);">${it.price_action}</span></td>
          <td class="cell-nowrap"><button class="btn-fly-3d">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          if (itemEntity) focusEntityIn3D(itemEntity);
        });
        tbodyInv.appendChild(tr);
      });
    }
    renderInventory();

    document.getElementById('search-inventory').addEventListener('input', (e) => {
      invSearch = e.target.value;
      renderInventory();
    });

    document.querySelectorAll('#inv-status-filters .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#inv-status-filters .toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        invFilterStatus = btn.getAttribute('data-status');
        renderInventory();
      });
    });

    // ----------------------------------------------------
    // TAB 5: Seasonal Demand Forecast (By Month & By Season)
    // ----------------------------------------------------
    const subtabMonthBtn = document.getElementById('subtab-btn-month');
    const subtabSeasonBtn = document.getElementById('subtab-btn-season');
    const containerMonth = document.getElementById('container-seasonal-month');
    const containerSeason = document.getElementById('container-seasonal-season');

    let currentSeasonalSubtab = 'month';
    function switchSeasonalSubView(view) {
      currentSeasonalSubtab = view;
      subtabMonthBtn.classList.toggle('active', view === 'month');
      subtabSeasonBtn.classList.toggle('active', view === 'season');
      containerMonth.style.display = view === 'month' ? 'flex' : 'none';
      containerSeason.style.display = view === 'season' ? 'flex' : 'none';
    }

    subtabMonthBtn.addEventListener('click', () => switchSeasonalSubView('month'));
    subtabSeasonBtn.addEventListener('click', () => switchSeasonalSubView('season'));

    // --- 5A: BY MONTH FORECAST ---
    const MONTHS = [
      { num: 1, abbr: 'JAN', name: 'January' },
      { num: 2, abbr: 'FEB', name: 'February' },
      { num: 3, abbr: 'MAR', name: 'March' },
      { num: 4, abbr: 'APR', name: 'April' },
      { num: 5, abbr: 'MAY', name: 'May' },
      { num: 6, abbr: 'JUN', name: 'June' },
      { num: 7, abbr: 'JUL', name: 'July' },
      { num: 8, abbr: 'AUG', name: 'August' },
      { num: 9, abbr: 'SEP', name: 'September' },
      { num: 10, abbr: 'OCT', name: 'October' },
      { num: 11, abbr: 'NOV', name: 'November' },
      { num: 12, abbr: 'DEC', name: 'December' }
    ];

    // Compute month units totals
    const monthUnitsMap = {};
    seasonalMonthly.forEach(it => {
      monthUnitsMap[it.month_number] = (monthUnitsMap[it.month_number] || 0) + (it.month_total_qty || 0);
    });

    let selectedMonth = 9; // Default to September
    let monthFilterTier = 'ALL';
    let monthSearchQuery = '';
    let monthSort = 'vel_desc';

    const monthGrid = document.getElementById('month-selector-grid');
    monthGrid.innerHTML = '';
    MONTHS.forEach(m => {
      const btn = document.createElement('div');
      btn.className = `month-pill-btn ${m.num === selectedMonth ? 'active' : ''}`;
      btn.setAttribute('data-month', m.num);
      const units = Math.round(monthUnitsMap[m.num] || 0);
      btn.innerHTML = `
        <div class="month-pill-abbr">${m.abbr}</div>
        <div style="font-size:0.7rem; color:var(--text-muted);">${m.name}</div>
        <div class="month-pill-units">${units.toLocaleString()} pcs</div>
      `;
      btn.addEventListener('click', () => {
        document.querySelectorAll('#month-selector-grid .month-pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedMonth = m.num;
        renderMonthForecast();
      });
      monthGrid.appendChild(btn);
    });

    const tbodyMonth = document.getElementById('tbody-seasonal-month');

    function renderMonthForecast() {
      const records = seasonalMonthly.filter(it => it.month_number === selectedMonth);

      // KPI stats
      const totalUnits = records.reduce((s, it) => s + (it.month_total_qty || 0), 0);
      const activeItemsCount = records.length;
      const peakCount = records.filter(it => it.demand_tier === 'PEAK_DEMAND' || it.monthly_velocity_index >= 1.5).length;
      const avgVel = records.length ? (records.reduce((s, it) => s + (it.monthly_velocity_index || 0), 0) / records.length) : 0;

      document.getElementById('kpi-month-units').textContent = `${Math.round(totalUnits).toLocaleString()} pcs`;
      document.getElementById('kpi-month-items').textContent = `${activeItemsCount.toLocaleString()} items`;
      document.getElementById('kpi-month-peak').textContent = `${peakCount.toLocaleString()} products`;
      document.getElementById('kpi-month-avg-vel').textContent = `${avgVel.toFixed(2)}x`;

      // Filter
      let filtered = records.filter(it => {
        const matchesTier = monthFilterTier === 'ALL' || it.demand_tier === monthFilterTier;
        let matchesSearch = true;
        if (monthSearchQuery) {
          const q = monthSearchQuery.toLowerCase();
          matchesSearch =
            (it.code && it.code.toLowerCase().includes(q)) ||
            (it.name && it.name.toLowerCase().includes(q)) ||
            (it.category && it.category.toLowerCase().includes(q));
        }
        return matchesTier && matchesSearch;
      });

      // Sort
      if (monthSort === 'vel_desc') {
        filtered.sort((a, b) => (b.monthly_velocity_index || 0) - (a.monthly_velocity_index || 0));
      } else if (monthSort === 'qty_desc') {
        filtered.sort((a, b) => (b.month_total_qty || 0) - (a.month_total_qty || 0));
      } else if (monthSort === 'stock_desc') {
        filtered.sort((a, b) => (b.current_stock || 0) - (a.current_stock || 0));
      } else if (monthSort === 'price_desc') {
        filtered.sort((a, b) => (b.price || 0) - (a.price || 0));
      }

      tbodyMonth.innerHTML = '';
      filtered.slice(0, 250).forEach(it => {
        let badgeHtml = '<span class="badge-status status-balanced">✓ NORMAL</span>';
        if (it.demand_tier === 'PEAK_DEMAND' || it.monthly_velocity_index >= 1.5) {
          badgeHtml = '<span class="badge-status status-peak">🔥 PEAK DEMAND</span>';
        } else if (it.demand_tier === 'HIGH_DEMAND' || it.monthly_velocity_index >= 1.2) {
          badgeHtml = '<span class="badge-status status-high">⚡ HIGH DEMAND</span>';
        } else if (it.demand_tier === 'LOW_SEASON' || it.monthly_velocity_index < 0.8) {
          badgeHtml = '<span class="badge-status status-off">❄️ OFF-PEAK</span>';
        }

        const velColor = it.monthly_velocity_index >= 1.5 ? 'var(--accent-rose)' :
                         it.monthly_velocity_index >= 1.2 ? 'var(--accent-amber)' :
                         it.monthly_velocity_index >= 0.8 ? 'var(--accent-emerald)' : 'var(--text-muted)';

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span style="font-family:var(--font-mono); color:var(--accent-cyan); font-weight:700;">${it.code}</span></td>
          <td style="font-weight:600; direction:rtl; text-align:right;">${it.name}</td>
          <td>${it.category}</td>
          <td style="font-family:var(--font-mono);">$${(it.price || 0).toFixed(2)}</td>
          <td style="font-family:var(--font-mono); font-weight:700;">${it.current_stock || 0}</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#fff;">${Math.round(it.month_total_qty || 0).toLocaleString()} pcs</td>
          <td>
            <span style="font-family:var(--font-mono); font-weight:700; color:${velColor}; font-size:1.05rem;">
              ${(it.monthly_velocity_index || 1).toFixed(2)}x
            </span>
          </td>
          <td>${badgeHtml}</td>
          <td><button class="btn-fly-3d">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          const itemEntity = itemMap.get(String(it.item_id));
          if (itemEntity) focusEntityIn3D(itemEntity);
        });
        tbodyMonth.appendChild(tr);
      });
    }

    renderMonthForecast();

    document.getElementById('search-seasonal-month').addEventListener('input', (e) => {
      monthSearchQuery = e.target.value;
      renderMonthForecast();
    });

    document.querySelectorAll('#month-tier-filters .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#month-tier-filters .toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        monthFilterTier = btn.getAttribute('data-tier');
        renderMonthForecast();
      });
    });

    document.getElementById('sort-seasonal-month').addEventListener('change', (e) => {
      monthSort = e.target.value;
      renderMonthForecast();
    });

    // --- 5B: BY SEASON FORECAST ---
    const SEASONS = [
      { id: 'Winter', icon: '❄️', title: 'Winter', months: 'Dec, Jan, Feb' },
      { id: 'Spring', icon: '🌸', title: 'Spring', months: 'Mar, Apr, May' },
      { id: 'Summer', icon: '☀️', title: 'Summer', months: 'Jun, Jul, Aug' },
      { id: 'Fall', icon: '🍁', title: 'Fall', months: 'Sep, Oct, Nov' }
    ];

    // Compute season totals
    const seasonSummary = {};
    SEASONS.forEach(s => {
      const sRecords = seasonalPeriod.filter(it => it.season === s.id);
      const sUnits = sRecords.reduce((sum, it) => sum + (it.season_total_qty || 0), 0);
      const sPeak = sRecords.filter(it => it.season_tier === 'PEAK_SEASON' || it.seasonal_velocity_index >= 1.3).length;
      
      const catCount = {};
      sRecords.forEach(it => {
        catCount[it.category] = (catCount[it.category] || 0) + (it.season_total_qty || 0);
      });
      let topCat = 'General';
      let topCatUnits = 0;
      Object.entries(catCount).forEach(([cat, q]) => {
        if (q > topCatUnits) {
          topCatUnits = q;
          topCat = cat;
        }
      });

      seasonSummary[s.id] = { units: sUnits, peakItems: sPeak, topCategory: topCat, count: sRecords.length };
    });

    let selectedSeason = 'Fall';
    let seasonFilterTier = 'ALL';
    let seasonSearchQuery = '';
    let seasonSort = 'vel_desc';

    const seasonCardsGrid = document.getElementById('season-cards-grid');
    seasonCardsGrid.innerHTML = '';
    SEASONS.forEach(s => {
      const summary = seasonSummary[s.id] || { units: 0, peakItems: 0, topCategory: 'N/A' };
      const card = document.createElement('div');
      card.className = `season-card ${s.id === selectedSeason ? 'active' : ''}`;
      card.setAttribute('data-season', s.id);
      card.innerHTML = `
        <div class="season-card-header">
          <div class="season-card-title">
            <span>${s.icon}</span>
            <span>${s.title}</span>
          </div>
          <span class="season-card-months">${s.months}</span>
        </div>
        <div class="season-card-stat">
          <span class="season-stat-label">Multi-Year Units</span>
          <span class="season-stat-val">${Math.round(summary.units).toLocaleString()} pcs</span>
        </div>
        <div style="font-size:0.72rem; color:var(--text-muted); display:flex; justify-content:space-between; margin-top:2px;">
          <span>Peak Surge: <strong style="color:var(--accent-rose);">${summary.peakItems}</strong></span>
          <span>Top: <strong style="color:var(--accent-cyan);">${summary.topCategory}</strong></span>
        </div>
      `;
      card.addEventListener('click', () => {
        document.querySelectorAll('#season-cards-grid .season-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        selectedSeason = s.id;
        renderSeasonForecast();
      });
      seasonCardsGrid.appendChild(card);
    });

    const tbodySeason = document.getElementById('tbody-seasonal-season');

    function renderSeasonForecast() {
      const records = seasonalPeriod.filter(it => it.season === selectedSeason);

      const totalUnits = records.reduce((s, it) => s + (it.season_total_qty || 0), 0);
      const peakCount = records.filter(it => it.season_tier === 'PEAK_SEASON' || it.seasonal_velocity_index >= 1.3).length;
      const avgVel = records.length ? (records.reduce((s, it) => s + (it.seasonal_velocity_index || 0), 0) / records.length) : 0;
      const topCat = (seasonSummary[selectedSeason] && seasonSummary[selectedSeason].topCategory) || 'General';

      document.getElementById('kpi-season-units').textContent = `${Math.round(totalUnits).toLocaleString()} pcs`;
      document.getElementById('kpi-season-peak').textContent = `${peakCount.toLocaleString()} products`;
      document.getElementById('kpi-season-cat').textContent = topCat;
      document.getElementById('kpi-season-avg-vel').textContent = `${avgVel.toFixed(2)}x`;

      let filtered = records.filter(it => {
        const matchesTier = seasonFilterTier === 'ALL' || it.season_tier === seasonFilterTier;
        let matchesSearch = true;
        if (seasonSearchQuery) {
          const q = seasonSearchQuery.toLowerCase();
          matchesSearch =
            (it.code && it.code.toLowerCase().includes(q)) ||
            (it.name && it.name.toLowerCase().includes(q)) ||
            (it.category && it.category.toLowerCase().includes(q));
        }
        return matchesTier && matchesSearch;
      });

      if (seasonSort === 'vel_desc') {
        filtered.sort((a, b) => (b.seasonal_velocity_index || 0) - (a.seasonal_velocity_index || 0));
      } else if (seasonSort === 'qty_desc') {
        filtered.sort((a, b) => (b.season_total_qty || 0) - (a.season_total_qty || 0));
      } else if (seasonSort === 'stock_desc') {
        filtered.sort((a, b) => (b.current_stock || 0) - (a.current_stock || 0));
      } else if (seasonSort === 'price_desc') {
        filtered.sort((a, b) => (b.price || 0) - (a.price || 0));
      }

      tbodySeason.innerHTML = '';
      filtered.slice(0, 250).forEach(it => {
        let badgeHtml = '<span class="badge-status status-balanced">⚖️ BALANCED</span>';
        if (it.season_tier === 'PEAK_SEASON' || it.seasonal_velocity_index >= 1.3) {
          badgeHtml = '<span class="badge-status status-peak">🔥 PEAK SEASON</span>';
        } else if (it.season_tier === 'HIGH_SEASON' || it.seasonal_velocity_index >= 1.1) {
          badgeHtml = '<span class="badge-status status-high">⚡ HIGH SEASON</span>';
        } else if (it.season_tier === 'OFF_SEASON' || it.seasonal_velocity_index < 0.85) {
          badgeHtml = '<span class="badge-status status-off">💤 OFF SEASON</span>';
        }

        const velColor = it.seasonal_velocity_index >= 1.3 ? 'var(--accent-rose)' :
                         it.seasonal_velocity_index >= 1.1 ? 'var(--accent-amber)' :
                         it.seasonal_velocity_index >= 0.85 ? 'var(--accent-emerald)' : 'var(--text-muted)';

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span style="font-family:var(--font-mono); color:var(--accent-cyan); font-weight:700;">${it.code}</span></td>
          <td style="font-weight:600; direction:rtl; text-align:right;">${it.name}</td>
          <td>${it.category}</td>
          <td style="font-family:var(--font-mono);">$${(it.price || 0).toFixed(2)}</td>
          <td style="font-family:var(--font-mono); font-weight:700;">${it.current_stock || 0}</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#fff;">${Math.round(it.season_total_qty || 0).toLocaleString()} pcs</td>
          <td>
            <span style="font-family:var(--font-mono); font-weight:700; color:${velColor}; font-size:1.05rem;">
              ${(it.seasonal_velocity_index || 1).toFixed(2)}x
            </span>
          </td>
          <td>${badgeHtml}</td>
          <td><button class="btn-fly-3d">🌐 3D</button></td>
        `;
        tr.querySelector('.btn-fly-3d').addEventListener('click', () => {
          const itemEntity = itemMap.get(String(it.item_id));
          if (itemEntity) focusEntityIn3D(itemEntity);
        });
        tbodySeason.appendChild(tr);
      });
    }

    renderSeasonForecast();

    document.getElementById('search-seasonal-season').addEventListener('input', (e) => {
      seasonSearchQuery = e.target.value;
      renderSeasonForecast();
    });

    document.querySelectorAll('#season-tier-filters .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#season-tier-filters .toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        seasonFilterTier = btn.getAttribute('data-tier');
        renderSeasonForecast();
      });
    });

    document.getElementById('sort-seasonal-season').addEventListener('change', (e) => {
      seasonSort = e.target.value;
      renderSeasonForecast();
    });
  }

  function setupEventListeners() {
    window.addEventListener('resize', onWindowResize);

    const container = renderer.domElement;
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('click', onMouseClick);
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  function onMouseMove(event) {
    if (activeTab !== 'tab-3d') return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(particleSystem);

    const tooltip = document.getElementById('hover-tooltip');
    if (intersects.length > 0) {
      const idx = intersects[0].index;
      const entity = allEntities[idx];
      if (entity) {
        document.body.style.cursor = 'pointer';
        tooltip.style.display = 'block';
        tooltip.style.left = `${event.clientX}px`;
        tooltip.style.top = `${event.clientY}px`;

        if (entity.type === 'item') {
          tooltip.innerHTML = `
            <div style="font-weight:700; color:#fff;">${entity.name || 'Item ' + entity.code}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono); margin-top:2px;">
              Code: <span style="color:var(--accent-cyan);">${entity.code}</span> • $${entity.price.toFixed(2)} • Stock: ${entity.stock}
            </div>
          `;
        } else if (entity.type === 'customer') {
          tooltip.innerHTML = `
            <div style="font-weight:700; color:var(--accent-gold);">👤 ${entity.name}</div>
            <div style="font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono); margin-top:2px;">
              Account #${entity.account_number} • Balance: $${entity.balance.toFixed(2)}
            </div>
          `;
        }
        return;
      }
    }

    document.body.style.cursor = 'default';
    tooltip.style.display = 'none';
  }

  function onMouseClick(event) {
    if (activeTab !== 'tab-3d' || event.target.tagName !== 'CANVAS') return;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(particleSystem);

    if (intersects.length > 0) {
      const idx = intersects[0].index;
      const entity = allEntities[idx];
      if (entity) {
        selectEntity(entity, true);
      }
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    TWEEN.update();
    controls.update();

    if (starfield) {
      starfield.rotation.y += 0.0002;
    }

    renderer.render(scene, camera);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
