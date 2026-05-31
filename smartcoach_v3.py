
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch chua duoc cai dat. Dung: pip install torch")
    print("[WARN] Se fallback ve LSTM numpy (random weights).\n")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[WARN] anthropic chua duoc cai dat. Dung: pip install anthropic")
    print("[WARN] Se fallback ve coaching template.\n")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
COURT_W, COURT_H  = 94, 50
SAMPLE_RATE       = 25
POSSESSION_SECS   = 14
N_FRAMES          = POSSESSION_SECS * SAMPLE_RATE   # 350
N_PLAYERS         = 10
INPUT_DIM         = N_PLAYERS * 5                   # 50
HIDDEN_DIM        = 64
TACTIC_LABELS     = ["Pick-and-Roll", "Isolation", "Motion Offense", "Fast Break", "Post-Up"]
N_TACTICS         = len(TACTIC_LABELS)
np.random.seed(42)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def smooth(arr, w=11):
    k = np.exp(-0.5 * np.linspace(-2, 2, w) ** 2); k /= k.sum(); p = w // 2
    out = np.zeros_like(arr)
    for d in range(arr.shape[1]):
        col = np.pad(arr[:, d], p, mode='edge')
        out[:, d] = np.convolve(col, k, mode='valid')
    return out

def micro_move(n, center, r=1.5, freq=0.7):
    t = np.linspace(0, 2 * np.pi * freq, n)
    px, py = np.random.uniform(0, 2 * np.pi, 2)
    x = center[0] + r * np.sin(t + px) + np.random.randn(n) * 0.2
    y = center[1] + r * np.cos(t * 1.3 + py) + np.random.randn(n) * 0.2
    return smooth(np.stack([x, y], axis=1))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_possession(tactic, n_frames=N_FRAMES):
    pos = np.zeros((n_frames, N_PLAYERS, 2))
    bx  = np.zeros(n_frames)
    by  = np.zeros(n_frames)

    if tactic == "Pick-and-Roll":
        ph1, ph2 = int(n_frames * .30), int(n_frames * .55)
        hx = np.concatenate([np.linspace(20, 35, ph1), np.linspace(35, 42, ph2 - ph1), np.linspace(42, 72, n_frames - ph2)])
        hy = np.concatenate([np.full(ph1, 25) + np.random.randn(ph1) * .3, np.linspace(25, 22, ph2 - ph1), np.linspace(22, 25, n_frames - ph2) + np.random.randn(n_frames - ph2) * .4])
        pos[:, 0, :] = smooth(np.stack([hx, hy], axis=1))
        sx = np.concatenate([np.full(ph1, 38) + np.random.randn(ph1) * .2, np.full(ph2 - ph1, 40), np.linspace(40, 74, n_frames - ph2)])
        sy = np.concatenate([np.full(ph1, 22) + np.random.randn(ph1) * .2, np.full(ph2 - ph1, 22), np.linspace(22, 25, n_frames - ph2)])
        pos[:, 1, :] = smooth(np.stack([sx, sy], axis=1))
        for p, spot in enumerate([[65, 8], [65, 42], [52, 5]]): pos[:, p + 2, :] = micro_move(n_frames, spot)
        pos[:, 5, :] = pos[:, 0, :] + np.random.randn(n_frames, 2) * 1.0 + [2, -1]
        pos[:, 6, :] = pos[:, 1, :] + np.random.randn(n_frames, 2) * 0.8 + [-1, 1]
        for p, spot in enumerate([[60, 12], [60, 38], [50, 25]]): pos[:, p + 7, :] = micro_move(n_frames, spot, r=2.0)
        pf = int(n_frames * .75)
        bx[:pf] = pos[:pf, 0, 0] + np.random.randn(pf) * .3; by[:pf] = pos[:pf, 0, 1] + np.random.randn(pf) * .3
        bx[pf:] = pos[pf:, 1, 0] + np.random.randn(n_frames - pf) * .3; by[pf:] = pos[pf:, 1, 1] + np.random.randn(n_frames - pf) * .3

    elif tactic == "Isolation":
        ph1 = int(n_frames * .4)
        ix = np.concatenate([np.linspace(30, 38, ph1) + np.random.randn(ph1) * .3, np.linspace(38, 75, n_frames - ph1)])
        iy = np.concatenate([25 + 3 * np.sin(np.linspace(0, 2 * np.pi, ph1)), np.linspace(25, 23, n_frames - ph1) + np.random.randn(n_frames - ph1) * .4])
        pos[:, 0, :] = smooth(np.stack([ix, iy], axis=1))
        for p, spot in enumerate([[62, 6], [62, 44], [50, 4], [50, 46]]): pos[:, p + 1, :] = micro_move(n_frames, spot, r=1.0, freq=.5)
        pos[:, 5, :] = pos[:, 0, :] + np.column_stack([np.random.randn(n_frames) * .8 + 1.5, np.random.randn(n_frames) * .8])
        for p, spot in enumerate([[58, 8], [58, 42], [48, 5], [48, 46]]): pos[:, p + 6, :] = micro_move(n_frames, spot)
        bx = pos[:, 0, 0] + np.random.randn(n_frames) * .3; by = pos[:, 0, 1] + np.random.randn(n_frames) * .3

    elif tactic == "Motion Offense":
        perim  = [[42, 25], [50, 8], [62, 5], [68, 42], [55, 44]]
        frames = [0, 70, 140, 210, 280, n_frames]
        holder = [0, 1, 2, 3, 4, 0]
        for p in range(5):
            base = np.array(perim[p], dtype=float); traj = micro_move(n_frames, base, r=3.5, freq=.6)
            cs, ce = int(n_frames * p / 5), int(n_frames * p / 5) + int(n_frames * .15)
            if ce < n_frames:
                traj[cs:ce, 0] = np.linspace(base[0], 72, ce - cs)
                traj[cs:ce, 1] = np.linspace(base[1], 25 + (p - 2) * 4, ce - cs)
            pos[:, p, :] = traj; pos[:, p + 5, :] = traj + np.random.randn(n_frames, 2) * 1.0 + [1, 0]
        for i, (s, e) in enumerate(zip(frames[:-1], frames[1:])):
            e = min(e, n_frames); h = holder[i]
            bx[s:e] = pos[s:e, h, 0] + np.random.randn(e - s) * .3
            by[s:e] = pos[s:e, h, 1] + np.random.randn(e - s) * .3

    elif tactic == "Fast Break":
        for p, ly in enumerate([20, 25, 30]):
            spd = np.power(np.linspace(0, 1, n_frames), .7)
            pos[:, p, 0] = smooth((5 + spd * 82).reshape(-1, 1), 7).ravel()
            pos[:, p, 1] = smooth((ly + np.random.randn(n_frames) * .5).reshape(-1, 1), 7).ravel()
        for p in range(2):
            pos[:, p + 3, 0] = smooth((np.linspace(5, 55, n_frames) + np.random.randn(n_frames) * .4).reshape(-1, 1), 9).ravel()
            pos[:, p + 3, 1] = smooth((15 + p * 20 + np.random.randn(n_frames) * .6).reshape(-1, 1), 9).ravel()
        for p in range(5):
            pos[:, p + 5, 0] = smooth((np.linspace(85, 65, n_frames) + np.random.randn(n_frames) * 1.5).reshape(-1, 1), 7).ravel()
            pos[:, p + 5, 1] = smooth((10 + p * 8 + np.random.randn(n_frames) * 1.0).reshape(-1, 1), 7).ravel()
        bx = pos[:, 0, 0] + np.random.randn(n_frames) * .3; by = pos[:, 0, 1] + np.random.randn(n_frames) * .3

    elif tactic == "Post-Up":
        px2 = 60 + 3 * np.sin(np.linspace(0, 2 * np.pi * .8, n_frames))
        py2 = 25 + 2 * np.cos(np.linspace(0, 2 * np.pi * 1.1, n_frames))
        pos[:, 0, :] = smooth(np.stack([px2, py2], axis=1))
        pos[:, 5, :] = pos[:, 0, :] + np.column_stack([np.random.randn(n_frames) * .8 - 1, np.random.randn(n_frames) * .8])
        for p, spot in enumerate([[50, 8], [50, 42], [42, 10], [42, 40], [35, 25]]): pos[:, p + 1, :] = micro_move(n_frames, spot, r=2.0, freq=.7)
        for p, spot in enumerate([[48, 10], [48, 40], [40, 12], [40, 38]]): pos[:, p + 6, :] = micro_move(n_frames, spot, r=1.8)
        bx = pos[:, 0, 0] + np.random.randn(n_frames) * .4; by = pos[:, 0, 1] + np.random.randn(n_frames) * .4

    pos  = np.clip(pos, [0, 0], [COURT_W, COURT_H])
    ball = np.clip(np.stack([bx, by], axis=1), [0, 0], [COURT_W, COURT_H])
    return {"positions": pos, "ball": ball, "tactic": tactic}


def build_sequence(data):
    """Chuyen possession data thanh sequence features (T, INPUT_DIM)."""
    pos = data["positions"]; ball = data["ball"]; T = len(pos)
    feats = []
    for t in range(T):
        f = []
        for p in range(N_PLAYERS):
            f.extend([pos[t, p, 0] / COURT_W, pos[t, p, 1] / COURT_H])
            f.append(np.linalg.norm(pos[t, p] - ball[t]) / COURT_W)
            vel = (pos[t, p] - pos[t - 1, p]) * SAMPLE_RATE if t > 0 else np.zeros(2)
            f.extend([vel[0] / 30, vel[1] / 30])
        feats.append(f)
    return np.array(feats, dtype=np.float32)  # (T, INPUT_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LSTM ENCODER — PyTorch (trained) hoac NumPy fallback
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. PyTorch LSTM ──────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        """
        LSTM encoder + classification head.
        forward() tra ve (logits, hidden_vector).
        """
        def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                     num_layers=2, n_classes=N_TACTICS, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.head    = nn.Linear(hidden_dim, n_classes)

        def forward(self, x):
            # x: (batch, T, input_dim)
            _, (h_n, _) = self.lstm(x)
            vec = h_n[-1]                   # last layer hidden state: (batch, hidden_dim)
            logits = self.head(self.dropout(vec))
            return logits, vec              # (batch, n_classes), (batch, hidden_dim)

        def encode(self, x):
            """Chi lay vector, khong lay logits."""
            _, vec = self.forward(x)
            return vec


class LSTMEncoder:
    """
    Wrapper: neu PyTorch co san -> train LSTM that su.
    Nguoc lai -> fallback ve NumPy LSTM voi weight ngau nhien (nhu v2).
    """

    def __init__(self):
        self.trained = False
        self.le      = LabelEncoder().fit(TACTIC_LABELS)
        if TORCH_AVAILABLE:
            self.net = LSTMNet()
        else:
            # NumPy fallback (v2 behavior)
            self._init_numpy_lstm()

    # ── NumPy fallback ────────────────────────────────────────────────────────
    def _init_numpy_lstm(self):
        rng = np.random.default_rng(42); sc = np.sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM))
        self.Wf = rng.normal(0, sc, (HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM)); self.bf = np.zeros(HIDDEN_DIM)
        self.Wi = rng.normal(0, sc, (HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM)); self.bi = np.zeros(HIDDEN_DIM)
        self.Wo = rng.normal(0, sc, (HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM)); self.bo = np.zeros(HIDDEN_DIM)
        self.Wc = rng.normal(0, sc, (HIDDEN_DIM, INPUT_DIM + HIDDEN_DIM)); self.bc = np.zeros(HIDDEN_DIM)

    def _numpy_forward(self, X):
        sig  = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        tanh = lambda x: np.tanh(np.clip(x, -500, 500))
        h = np.zeros(HIDDEN_DIM); c = np.zeros(HIDDEN_DIM)
        for t in range(len(X)):
            z = np.concatenate([X[t], h])
            f = sig(self.Wf @ z + self.bf); i = sig(self.Wi @ z + self.bi)
            o = sig(self.Wo @ z + self.bo); ct = tanh(self.Wc @ z + self.bc)
            c = f * c + i * ct; h = o * tanh(c)
        return h

    # ── Training (chỉ chạy khi có PyTorch) ───────────────────────────────────
    def train_model(self, sequences, labels,
                    epochs=15, batch_size=16, lr=1e-3, val_ratio=0.2):
        """
        Train LSTM supervised.
        sequences: list of np.array (T, INPUT_DIM)
        labels   : list of str
        """
        if not TORCH_AVAILABLE:
            print("  [SKIP] PyTorch khong co san, bo qua buoc train LSTM.")
            return

        y = self.le.transform(labels)

        # Padding: tat ca sequence cung chieu dai
        max_T = max(s.shape[0] for s in sequences)
        X_pad = np.zeros((len(sequences), max_T, INPUT_DIM), dtype=np.float32)
        for i, s in enumerate(sequences):
            X_pad[i, :s.shape[0], :] = s

        X_t = torch.tensor(X_pad)
        y_t = torch.tensor(y, dtype=torch.long)

        # Train/val split
        n_val   = max(1, int(len(X_t) * val_ratio))
        idx     = torch.randperm(len(X_t))
        tr_idx  = idx[n_val:]
        val_idx = idx[:n_val]

        tr_ds  = TensorDataset(X_t[tr_idx], y_t[tr_idx])
        val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])
        tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state   = None

        for ep in range(1, epochs + 1):
            self.net.train()
            tr_loss = 0
            for Xb, yb in tr_dl:
                optimizer.zero_grad()
                logits, _ = self.net(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
            scheduler.step()

            # Validation
            self.net.eval()
            correct = total = 0
            with torch.no_grad():
                for Xb, yb in val_dl:
                    logits, _ = self.net(Xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in self.net.state_dict().items()}

            if ep % 5 == 0 or ep == 1:
                print(f"  Epoch {ep:02d}/{epochs}  loss={tr_loss/len(tr_dl):.4f}  val_acc={val_acc*100:.1f}%")

        # Restore best weights
        if best_state:
            self.net.load_state_dict(best_state)
        print(f"  => Best val accuracy: {best_val_acc*100:.1f}%")
        self.trained = True

    # ── Encode một possession ─────────────────────────────────────────────────
    def encode_possession(self, data):
        seq = build_sequence(data)   # (T, INPUT_DIM)
        if TORCH_AVAILABLE:
            self.net.eval()
            with torch.no_grad():
                x = torch.tensor(seq).unsqueeze(0)   # (1, T, INPUT_DIM)
                vec = self.net.encode(x).squeeze(0).numpy()
        else:
            vec = self._numpy_forward(seq)
        return vec   # (HIDDEN_DIM,)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIER — GB dung LSTM vec + context; EPV trained regression
# ═══════════════════════════════════════════════════════════════════════════════

# Coaching template — fallback khi khong co API key
COACHING_TEMPLATE = {
    "Pick-and-Roll":  {
        "strengths":   "Tao mismatch hieu qua, buoc doi thu switch",
        "when_best":   "Shot clock > 14s, guard nhanh hon center doi thu",
        "adjustment":  "Drop coverage: screener pull-up | Hard hedge: ball-handler drive",
        "drill":       "2-man game: hand-off + mid-range decision"
    },
    "Isolation": {
        "strengths":   "Khai thac mismatch ca nhan, kiem soat tempo",
        "when_best":   "Cuoi hiep, shot clock < 8s, star player dang hot hand",
        "adjustment":  "Double-team: kick out corner 3 | Switch: exploit size",
        "drill":       "1-on-1 footwork: jab step + spin move finishing"
    },
    "Motion Offense": {
        "strengths":   "Kho defend, tao nhieu passing lane, met doi thu",
        "when_best":   "Doi tan cong co shooting tot o tat ca vi tri",
        "adjustment":  "Sag defense: spot-up 3 | Tight D: back-cut",
        "drill":       "5-on-0 motion: pass and cut, screen-away"
    },
    "Fast Break": {
        "strengths":   "Diem de nhat, doi thu chua set defense",
        "when_best":   "Sau turnover hoac defensive rebound, score diff <= 10",
        "adjustment":  "Defense tra ve kip: dung lai, to chuc half-court",
        "drill":       "3-on-2 -> 2-on-1 transition drill"
    },
    "Post-Up": {
        "strengths":   "Khai thac loi the the hinh, drawing fouls hieu qua",
        "when_best":   "Doi thu co center nho/yeu hoac foul trouble",
        "adjustment":  "Double-team: pass ra canh corner 3",
        "drill":       "Mikan drill + Drop step + Jump hook"
    },
}


def get_llm_coaching(tactic, epv, shot_clock, score_diff, quarter):
    """
    [FIX 3] Goi Anthropic API that su thay vi dung template cung.
    Fallback ve template neu khong co API key hoac loi mang.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ANTHROPIC_AVAILABLE or not api_key:
        tmpl = COACHING_TEMPLATE[tactic]
        return {
            "strengths":  tmpl["strengths"],
            "when_best":  tmpl["when_best"],
            "adjustment": tmpl["adjustment"],
            "drill":      tmpl["drill"],
            "source":     "template"
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        score_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)
        prompt = f"""Ban la HLV bong ro chuyen nghiep dang phan tich mot luot tan cong.

Chien thuat dang dung: {tactic}
Nguyen canh tran dau:
- Shot clock con lai: {shot_clock:.0f} giay
- Cach biet ti so: {score_str} (so voi doi thu)
- Hiep dang choi: {quarter}/4
- EPV (Expected Possession Value): {epv:.3f} (trung binh = 1.00)

Hay dua ra goi y ngan gon theo dung 4 dong, moi dong mot y:
DIEM MANH: [1 cau ngan ve loi the cua chien thuat nay trong tinh huong nay]
KHI NAO DUNG: [1 cau cu the cho tinh huong nay]
DIEU CHINH: [1 cau dieu chinh chien thuat dua vao EPV va score diff]
BAI TAP: [ten 1 bai tap luyen tap cu the]

Chi tra loi dung 4 dong, khong giai thich them."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        lines = {ln.split(":")[0].strip(): ":".join(ln.split(":")[1:]).strip()
                 for ln in text.splitlines() if ":" in ln}
        return {
            "strengths":  lines.get("DIEM MANH",  COACHING_TEMPLATE[tactic]["strengths"]),
            "when_best":  lines.get("KHI NAO DUNG", COACHING_TEMPLATE[tactic]["when_best"]),
            "adjustment": lines.get("DIEU CHINH", COACHING_TEMPLATE[tactic]["adjustment"]),
            "drill":      lines.get("BAI TAP",    COACHING_TEMPLATE[tactic]["drill"]),
            "source":     "llm"
        }
    except Exception as e:
        print(f"  [WARN] LLM API loi ({e}), dung fallback template.")
        tmpl = COACHING_TEMPLATE[tactic]
        return {**tmpl, "source": "template"}


class TacticClassifier:
    def __init__(self):
        # [FIX 2] GB nhan HIDDEN_DIM (64) + 4 context = 68 features
        self.gb  = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                              learning_rate=0.1, random_state=42)
        # [FIX 4] EPV regression rieng, train tren features thuc
        self.epv_model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                   learning_rate=0.1, random_state=42)
        self.le      = LabelEncoder()
        self.scaler  = StandardScaler()

    def _build_X(self, vecs, ctxs):
        """Noi LSTM vector + game context."""
        return np.hstack([np.array(vecs), np.array(ctxs)])   # (N, 68)

    def train(self, vecs, ctxs, labels, epv_targets=None):
        """
        [FIX 2] Train GB tren ca LSTM vec + context.
        [FIX 4] Train EPV regression neu co targets.
        [FIX 5] In classification_report tren test set.
        """
        X = self._build_X(vecs, ctxs)
        y = self.le.fit_transform(labels)

        # [FIX 5] Train/test split 80/20
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        self.gb.fit(X_tr, y_tr)

        # Danh gia tren test set
        y_pred = self.gb.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        print(f"\n  === Test Set Evaluation (20% held-out) ===")
        print(f"  Accuracy: {acc*100:.1f}%")
        print(classification_report(y_te, y_pred,
                                    target_names=self.le.classes_,
                                    digits=2))

        # [FIX 4] Train EPV regression
        if epv_targets is not None:
            Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(
                X, np.array(epv_targets), test_size=0.2, random_state=42)
            Xe_tr = self.scaler.transform(Xe_tr)
            Xe_te = self.scaler.transform(Xe_te)
            self.epv_model.fit(Xe_tr, ye_tr)
            epv_r2 = self.epv_model.score(Xe_te, ye_te)
            print(f"  EPV Regression R2 (test): {epv_r2:.3f}")

        self._epv_trained = epv_targets is not None

    def predict(self, vec, ctx):
        """
        [FIX 2] Dung ca LSTM vector + context de predict.
        [FIX 4] EPV tu regression model (neu da train), khong dung lookup+noise.
        """
        X = self.scaler.transform(
            np.hstack([vec, ctx]).reshape(1, -1)
        )
        idx    = self.gb.predict(X)[0]
        proba  = self.gb.predict_proba(X)[0]
        tactic = self.le.inverse_transform([idx])[0]

        # EPV: dung regression model neu co, fallback ve rule-based
        if hasattr(self, '_epv_trained') and self._epv_trained:
            epv = float(self.epv_model.predict(X)[0])
        else:
            epv_base = {
                "Pick-and-Roll": 1.05, "Fast Break": 1.18,
                "Motion Offense": 1.08, "Isolation": 0.92, "Post-Up": 1.02
            }
            # Van giu yeu to shot_clock (ctx[0]) nhung khong them noise ngau nhien
            epv = epv_base.get(tactic, 1.0) * (0.7 + 0.3 * ctx[0])

        return {
            "tactic":        tactic,
            "confidence":    float(proba.max()),
            "probabilities": dict(zip(self.le.classes_, proba.tolist())),
            "epv":           round(float(np.clip(epv, 0.6, 1.5)), 3)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COURT DRAWING
# ═══════════════════════════════════════════════════════════════════════════════
def draw_court(ax):
    ax.set_facecolor("#16213e")
    lc = "#c8d6e5"; lw = 1.2
    ax.add_patch(patches.Rectangle((0, 0), COURT_W, COURT_H, lw=2, ec=lc, fc="#16213e"))
    ax.add_patch(patches.Rectangle((75, 17), 19, 16, lw=lw, ec=lc, fc="#0f3460", alpha=.7))
    ax.add_patch(plt.Circle((75, 25), 6, color=lc, fill=False, lw=lw))
    ax.add_patch(patches.Arc((88, 25), 47, 47, angle=0, theta1=110, theta2=250, color=lc, lw=lw))
    ax.plot([75, 94], [4, 4], color=lc, lw=lw); ax.plot([75, 94], [46, 46], color=lc, lw=lw)
    ax.add_patch(plt.Circle((88, 25), .75, color="#ff6b35", lw=2, fill=False))
    ax.plot([88, 94], [25, 25], color=lc, lw=2)
    ax.add_patch(plt.Circle((0, 25), 6, color=lc, fill=False, lw=lw, linestyle="--", alpha=.4))
    ax.set_xlim(0, COURT_W); ax.set_ylim(0, COURT_H); ax.set_aspect("equal"); ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRADIENT TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════
def plot_gradient_trajectory(ax, x, y, cmap="plasma", lw=2.0, alpha=0.9, step=1):
    x2, y2 = x[::step], y[::step]
    pts  = np.array([x2, y2]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(segs)))
    lc_obj = LineCollection(segs, colors=cols, linewidth=lw, alpha=alpha)
    ax.add_collection(lc_obj)
    if len(x2) > 5:
        ax.annotate("", xy=(x2[-1], y2[-1]), xytext=(x2[-3], y2[-3]),
                    arrowprops=dict(arrowstyle="->", color=cols[-1], lw=1.5))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def visualize_dashboard(poss, result, encoder, coaching):
    fig = plt.figure(figsize=(22, 13), facecolor="#0d0d1a")
    fig.suptitle("SmartCoach AI v3  |  Basketball Tactical Intelligence  |  i-TECH",
                 fontsize=16, color="#ffffff", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=.4, wspace=.3,
                           left=.03, right=.97, top=.93, bottom=.04)

    pos  = poss["positions"]
    ball = poss["ball"]
    snap = 175

    # ── Panel 1: Court snapshot ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    draw_court(ax1)
    trail_len = 30
    for p in range(5):
        tx = pos[max(0, snap - trail_len):snap, p, 0]
        ty = pos[max(0, snap - trail_len):snap, p, 1]
        plot_gradient_trajectory(ax1, tx, ty, cmap="Oranges", lw=1.5, alpha=.6, step=1)
    p5 = pos[snap, :5, :]; p10 = pos[snap, 5:, :]; bpos = ball[snap]
    ax1.scatter(p5[:, 0], p5[:, 1], s=260, c="#ff6b35", ec="white", lw=1.5, zorder=6)
    ax1.scatter(p10[:, 0], p10[:, 1], s=260, c="#4ecdc4", marker="s", ec="white", lw=1.5, zorder=6)
    ax1.scatter([bpos[0]], [bpos[1]], s=130, c="#f9ca24", ec="#f0932b", lw=2, zorder=7)
    for i in range(5):
        ax1.text(p5[i, 0], p5[i, 1], str(i + 1), ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white", zorder=8)
    ax1.set_title(f"Snapshot t={snap // SAMPLE_RATE:.1f}s  |  {result['tactic']}",
                  color="white", fontsize=10, pad=6)
    leg1 = patches.Patch(color="#ff6b35", label="Attack")
    leg2 = patches.Patch(color="#4ecdc4", label="Defense")
    ax1.legend(handles=[leg1, leg2], loc="upper left",
               facecolor="#1a1a2e", labelcolor="white", fontsize=7, framealpha=.9)

    # ── Panel 2: Full trajectory ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    draw_court(ax2)
    for p, cmap in enumerate(["plasma", "viridis", "cool", "spring", "autumn"]):
        tx = pos[:, p, 0]; ty = pos[:, p, 1]
        plot_gradient_trajectory(ax2, tx, ty, cmap=cmap, lw=1.8, alpha=.75, step=2)
        ax2.scatter([tx[0]], [ty[0]], s=35, c="white", zorder=5, alpha=.8)
    plot_gradient_trajectory(ax2, ball[:, 0], ball[:, 1], cmap="YlOrRd", lw=2.5, alpha=.9, step=3)
    ax2.axvline(pos[int(N_FRAMES * .3), 0, 0], color="white", lw=.7, linestyle=":", alpha=.4)
    ax2.text(pos[int(N_FRAMES * .3), 0, 0] + .5, 3, "Screen", color="white", fontsize=6, alpha=.7)
    ax2.set_title("Trajectory (gradient = thoi gian, trang=bat dau, vang=ket thuc)",
                  color="white", fontsize=9, pad=6)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, POSSESSION_SECS))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=.025, pad=.02)
    cbar.set_label("Time (s)", color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=6)

    # ── Panel 3: LSTM fingerprint ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    vec = encoder.encode_possession(poss)
    trained_label = "(TRAINED)" if (TORCH_AVAILABLE and encoder.trained) else "(random weights)"
    im  = ax3.imshow(vec.reshape(8, 8), cmap="RdYlGn", aspect="auto", vmin=-1.5, vmax=1.5)
    plt.colorbar(im, ax=ax3, fraction=.046, pad=.04).ax.yaxis.set_tick_params(color="white")
    ax3.set_title(f"Tang 1: LSTM Vector 64-chieu {trained_label}\n(Tactical Fingerprint)",
                  color="white", fontsize=9, pad=6)
    ax3.tick_params(colors="white"); ax3.set_facecolor("#1a1a2e")

    # ── Panel 4: Probability bars ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    tactics = list(result["probabilities"].keys())
    probs   = list(result["probabilities"].values())
    order   = np.argsort(probs)
    tactics_s = [tactics[i] for i in order]; probs_s = [probs[i] for i in order]
    bar_colors = ["#ff6b35" if t == result["tactic"] else "#4ecdc4" for t in tactics_s]
    bars = ax4.barh(tactics_s, probs_s, color=bar_colors, ec="white", lw=.8, height=.55)
    for bar, p2 in zip(bars, probs_s):
        ax4.text(bar.get_width() + .01, bar.get_y() + bar.get_height() / 2,
                 f"{p2 * 100:.1f}%", va="center", color="white", fontsize=9)
    ax4.set_xlim(0, 1.15)
    ax4.set_title(f"Tang 2: Gradient Boosting (vec+ctx)\nChien thuat: {result['tactic']} ({result['confidence'] * 100:.0f}%)",
                  color="white", fontsize=10, pad=6)
    ax4.set_facecolor("#1a1a2e"); ax4.tick_params(colors="white")
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]: ax4.spines[sp].set_edgecolor("#555")

    # ── Panel 5: EPV gauge ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#1a1a2e"); ax5.axis("off")
    epv = result["epv"]
    x_gauge = np.linspace(.7, 1.3, 200)
    c_gauge = plt.cm.RdYlGn((x_gauge - .7) / .6)
    for xi in range(len(x_gauge) - 1):
        ax5.axvspan(x_gauge[xi], x_gauge[xi + 1], ymin=.3, ymax=.7, color=c_gauge[xi], alpha=.8)
    ax5.axvline(epv, color="white", lw=4, ymin=.15, ymax=.85, zorder=5)
    ax5.axvline(1.0, color="#aaa", lw=1.5, ymin=.15, ymax=.85, linestyle="--", alpha=.7)
    ax5.set_xlim(.65, 1.35); ax5.set_ylim(0, 1)
    ax5.text(epv, .88, f"{epv:.3f}", ha="center", color="white", fontsize=18, fontweight="bold")
    ax5.text(1.0, .12, "avg\n1.00", ha="center", color="#aaa", fontsize=7)
    ax5.text(.67, .5, "EPV", color="white", fontsize=12, va="center", fontweight="bold")
    epv_lbl = "TUYET VOI" if epv >= 1.05 else ("TRUNG BINH" if epv >= 0.95 else "RUI RO")
    epv_col = "#27ae60" if epv >= 1.05 else ("#f39c12" if epv >= 0.95 else "#e74c3c")
    ax5.text(.5, .5, epv_lbl, transform=ax5.transAxes, ha="center", va="center",
             fontsize=15, color=epv_col, fontweight="bold", alpha=.3)
    epv_src = "Regression Model" if (hasattr(TacticClassifier, '_epv_trained')) else "Rule-based"
    ax5.set_title(f"Expected Possession Value\n(Tich hop Game Context | {epv_src})",
                  color="white", fontsize=10, pad=6)

    # ── Panel 6: Coaching (LLM hoac template) ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#0d1b2a"); ax6.axis("off")
    src_label = "LLM (Anthropic API)" if coaching.get("source") == "llm" else "Template Fallback"
    lines = [
        (f"TANG 3: COACHING — {src_label}", "#ff6b35", 9, True),
        (f"Chien thuat: {result['tactic']}", "#ffffff", 10, True),
        ("", "#888", 8, False),
        ("DIEM MANH:", "#4ecdc4", 8, True),
        (f"  {coaching['strengths']}", "#dddddd", 8, False),
        ("", "#888", 8, False),
        ("KHI NAO DUNG:", "#4ecdc4", 8, True),
        (f"  {coaching['when_best']}", "#dddddd", 8, False),
        ("", "#888", 8, False),
        ("DIEU CHINH:", "#4ecdc4", 8, True),
        (f"  {coaching['adjustment'][:65]}", "#dddddd", 8, False),
        ("", "#888", 8, False),
        ("BAI TAP:", "#7bed9f", 8, True),
        (f"  {coaching['drill']}", "#dddddd", 8, False),
    ]
    y_pos = 0.97
    for txt, col, sz, bold in lines:
        ax6.text(.04, y_pos, txt, transform=ax6.transAxes, color=col, fontsize=sz,
                 va="top", fontweight="bold" if bold else "normal")
        y_pos -= 0.068
    ax6.set_title("Tang 3: LLM Coaching Layer", color="white", fontsize=10, pad=6)

    fig.text(.5, .005,
             "SmartCoach AI v3  |  i-TECH Research Group  |  LSTM(trained) -> GB(vec+ctx) -> LLM(API)",
             ha="center", color="#666", fontsize=7, style="italic")

    plt.savefig("smartcoach_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Dashboard saved: smartcoach_dashboard.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FINGERPRINT PLOT
# ═══════════════════════════════════════════════════════════════════════════════
def plot_fingerprints(vecs, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d0d1a")
    fig.suptitle("Tactical Fingerprint Analysis  |  SmartCoach AI v3",
                 color="white", fontsize=13, fontweight="bold")
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(np.array(vecs))
    colors = {
        "Pick-and-Roll": "#ff6b35", "Isolation": "#e056fd",
        "Motion Offense": "#4ecdc4", "Fast Break": "#f9ca24", "Post-Up": "#7bed9f"
    }
    ax1.set_facecolor("#1a1a2e")
    for t in TACTIC_LABELS:
        mask = np.array(labels) == t
        ax1.scatter(proj[mask, 0], proj[mask, 1], s=70, c=colors[t],
                    label=t, alpha=.85, ec="white", lw=.4)
        cx, cy = proj[mask, 0].mean(), proj[mask, 1].mean()
        ax1.scatter([cx], [cy], s=200, c=colors[t], ec="white", lw=2, marker="*", zorder=6)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=.9)
    ax1.set_title(
        f"PCA 64D->2D  (PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1] * 100:.1f}%)",
        color="white", fontsize=10)
    ax1.tick_params(colors="white"); ax1.set_xlabel("PC1", color="#aaa"); ax1.set_ylabel("PC2", color="#aaa")
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")

    mean_vecs = np.zeros((len(TACTIC_LABELS), HIDDEN_DIM))
    for i, t in enumerate(TACTIC_LABELS):
        mask = np.array(labels) == t
        mean_vecs[i] = np.array(vecs)[mask].mean(axis=0)
    im = ax2.imshow(mean_vecs, cmap="RdYlGn", aspect="auto", vmin=-1.5, vmax=1.5)
    ax2.set_yticks(range(len(TACTIC_LABELS)))
    ax2.set_yticklabels(TACTIC_LABELS, color="white", fontsize=9)
    ax2.set_xlabel("Vector Dimension (0-63)", color="#aaa")
    ax2.set_title("Mean Tactical Fingerprint per Play Type", color="white", fontsize=10)
    plt.colorbar(im, ax=ax2, label="activation", fraction=.03)
    ax2.tick_params(colors="white")

    fig.tight_layout()
    plt.savefig("tactical_fingerprints.png", dpi=130, bbox_inches="tight", facecolor="#0d0d1a")
    print("  Fingerprints saved: tactical_fingerprints.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  SmartCoach AI v3  |  i-TECH Research Group")
    print("  Cai tien: LSTM trained | GB(vec+ctx) | LLM API | EPV regression")
    print("=" * 65)

    # ── Sinh du lieu huấn luyện ─────────────────────────────────────────────
    print("\nTao du lieu huan luyen (40 possessions x 5 tactics = 200 samples)...",
          end=" ", flush=True)
    encoder = LSTMEncoder()
    sequences, vecs, ctxs, labels, epv_targets = [], [], [], [], []

    epv_base = {
        "Pick-and-Roll": 1.05, "Fast Break": 1.18,
        "Motion Offense": 1.08, "Isolation": 0.92, "Post-Up": 1.02
    }

    for tactic in TACTIC_LABELS:
        for _ in range(40):
            d  = generate_possession(tactic)
            sc = np.random.uniform(5, 24)
            sd = np.random.randint(-15, 16)
            q  = np.random.randint(1, 5)
            tf = np.random.randint(0, 6)
            ctx = [sc / 24, sd / 30, q / 4, tf / 5]

            seq = build_sequence(d)
            sequences.append(seq)
            ctxs.append(ctx)
            labels.append(tactic)

            # EPV target: base + shot_clock bonus + score pressure + noise
            epv = epv_base[tactic] * (0.7 + 0.3 * (sc / 24))
            epv += 0.05 * (sd / 30)          # dang thang thi EPV cao hon
            epv += np.random.normal(0, 0.02)  # noise nho
            epv_targets.append(float(np.clip(epv, 0.6, 1.5)))

    print(f"OK ({len(labels)} samples)")

    # ── [FIX 1] Train LSTM ──────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print("\n[TANG 1 - TRAIN LSTM (PyTorch)]")
        encoder.train_model(sequences, labels, epochs=15, batch_size=16)
    else:
        print("\n[TANG 1 - LSTM fallback (NumPy random weights)]")

    # Encode tat ca sau khi train
    for i, d_seq in enumerate(sequences):
        # Tao lai data dict de encode
        d = generate_possession(labels[i])
        vecs.append(encoder.encode_possession(d))

    # ── [FIX 2 + 4 + 5] Train Classifier + EPV ─────────────────────────────
    print("\n[TANG 2 - TRAIN GRADIENT BOOSTING + EPV REGRESSION]")
    clf = TacticClassifier()
    clf.train(vecs, ctxs, labels, epv_targets=epv_targets)

    # ── Demo possession ─────────────────────────────────────────────────────
    DEMO = "Pick-and-Roll"
    shot_clock  = 16.0
    score_diff  = -3
    quarter     = 4
    time_remain = 3

    print(f"\n{'─'*65}")
    print(f"  DEMO POSSESSION: {DEMO}")
    print(f"  Shot clock: {shot_clock}s | Score: {score_diff:+d} | Q{quarter} | {time_remain} min left")
    print(f"{'─'*65}")

    poss = generate_possession(DEMO)
    ctx  = np.array([shot_clock / 24, score_diff / 30, quarter / 4, time_remain / 5])

    print("\n[TANG 1 - LSTM ENCODE]")
    vec = encoder.encode_possession(poss)
    src = "trained" if (TORCH_AVAILABLE and encoder.trained) else "random weights"
    print(f"  Input : {N_FRAMES} frames x {N_PLAYERS} players x 5 features")
    print(f"  Output: vector 64-chieu  ||v||={np.linalg.norm(vec):.3f}  ({src})")

    print("\n[TANG 2 - GRADIENT BOOSTING (vec + ctx)]")
    result = clf.predict(vec, ctx)
    print(f"  Chien thuat : {result['tactic']}  (confidence: {result['confidence'] * 100:.1f}%)")
    print(f"  EPV         : {result['epv']:.3f} pts/possession")
    for t, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        print(f"    {'#' * int(p * 30):<30} {t:<20} {p * 100:.1f}%")

    print("\n[TANG 3 - LLM COACHING]")
    coaching = get_llm_coaching(
        tactic=result['tactic'], epv=result['epv'],
        shot_clock=shot_clock, score_diff=score_diff, quarter=quarter
    )
    src_lbl = "Anthropic API" if coaching.get("source") == "llm" else "Template fallback"
    print(f"  Nguon       : {src_lbl}")
    print(f"  Diem manh   : {coaching['strengths']}")
    print(f"  Khi nao dung: {coaching['when_best']}")
    print(f"  Dieu chinh  : {coaching['adjustment']}")
    print(f"  Bai tap     : {coaching['drill']}")

    print("\n[RENDER DASHBOARD & FINGERPRINTS]")
    visualize_dashboard(poss, result, encoder, coaching)
    plot_fingerprints(vecs, labels)

    print("\n" + "=" * 65)
    print("  HOAN TAT!  smartcoach_dashboard.png  +  tactical_fingerprints.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
