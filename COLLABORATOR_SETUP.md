# Collaborator Setup Guide

This guide explains how to add contributors to the **private model repository** (`mma-artifacts`) so they can access trained models and artifacts.

---

## 🔐 Private Repository Access

**Repository:** `Alanbenny123/mma-artifacts`  
**Visibility:** Private (only authorized collaborators can access)

---

## Adding Collaborators

### Method 1: Using GitHub CLI (Recommended)

**Add a collaborator with read access:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X PUT -f permission=read
```

**Add a collaborator with write access:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X PUT -f permission=write
```

**Add a collaborator with admin access:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X PUT -f permission=admin
```

**Permission Levels:**
- `read`: Can view and download models/releases
- `write`: Can push changes, create releases
- `admin`: Full control (add/remove collaborators, delete repo)

**Example:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/contributor-name -X PUT -f permission=read
```

### Method 2: Using GitHub Web Interface

1. Go to: `https://github.com/Alanbenny123/mma-artifacts/settings/access`
2. Click **"Invite a collaborator"**
3. Enter GitHub username
4. Select permission level:
   - **Read** - View and download
   - **Write** - Push changes
   - **Admin** - Full control
5. Click **"Add [username] to this repository"**

### Method 3: Check Current Collaborators

```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators --jq '.[].login'
```

---

## For Contributors: Getting Access

### Step 1: Request Access

Contact the repository owner to request collaborator access.

### Step 2: Accept Invitation

1. Check your email for the GitHub invitation
2. Or go to: `https://github.com/Alanbenny123/mma-artifacts`
3. Accept the invitation

### Step 3: Authenticate GitHub CLI

```powershell
gh auth login
```

### Step 4: Download Models

**Option A: Using the script**
```powershell
pwsh -File scripts\pull_models.ps1
```

**Option B: Manual download**
```powershell
# Create output directory
New-Item -ItemType Directory -Force temptrainedoutput | Out-Null

# Download behavior model
gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "best_behavior_model_fixed.pth" `
  -D temptrainedoutput

gh release download behavior-v1 `
  -R Alanbenny123/mma-artifacts `
  -p "behavior_checkpoint.pth" `
  -D temptrainedoutput
```

**Option C: Clone the repository (if you need code access)**
```powershell
gh repo clone Alanbenny123/mma-artifacts
cd mma-artifacts
```

---

## Verify Access

**Check if you can access the repository:**
```powershell
gh repo view Alanbenny123/mma-artifacts
```

**List available releases:**
```powershell
gh release list -R Alanbenny123/mma-artifacts
```

**View release details:**
```powershell
gh release view behavior-v1 -R Alanbenny123/mma-artifacts
```

---

## Removing Collaborators

**Remove a collaborator:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X DELETE
```

**Or via web interface:**
1. Go to: `https://github.com/Alanbenny123/mma-artifacts/settings/access`
2. Find the collaborator
3. Click **"Remove"** next to their name

---

## Troubleshooting

### "Permission denied" error

**Issue:** Contributor can't access the repository.

**Solutions:**
1. Verify they accepted the invitation (check email)
2. Verify their GitHub username is correct
3. Re-invite them if needed

### "Not authenticated" error

**Issue:** `gh` commands fail with authentication error.

**Solution:**
```powershell
gh auth login
```

### "Repository not found" error

**Issue:** Contributor sees "repository not found" even after invitation.

**Solutions:**
1. Ensure they accepted the invitation
2. Check their GitHub account is logged in
3. Verify repository name: `Alanbenny123/mma-artifacts`

---

## Security Notes

- ✅ Private repository is **not visible** to non-collaborators
- ✅ Releases and assets are **private** by default
- ✅ Only collaborators can download models
- ✅ Public repository (`mma`) does **not** expose private repo URLs or access

---

## Quick Reference

**Add collaborator (read):**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X PUT -f permission=read
```

**List collaborators:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators --jq '.[].login'
```

**Remove collaborator:**
```powershell
gh api repos/Alanbenny123/mma-artifacts/collaborators/<USERNAME> -X DELETE
```

**Download models (for contributors):**
```powershell
pwsh -File scripts\pull_models.ps1
```

---

**Last Updated:** 2024

