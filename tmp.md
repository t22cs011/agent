graph TD
    subgraph Host_Server_Sirius [Host Server: Sirius]
        style Host_Server_Sirius fill:#f9f9f9,stroke:#333,stroke-width:2px
        
        Agent_Env[("🐍 agent_env (Conda)<br>Active")] 
        style Agent_Env fill:#d4edda,stroke:#28a745,stroke-width:2px
        
        BCI2020[("💀 bci2020 (Conda)<br>Deprecated")]
        style BCI2020 fill:#f8d7da,stroke:#dc3545,stroke-dasharray: 5 5
        
        subgraph Docker_Runtime [Docker Runtime]
            BCI_Container[("🐳 bci (Docker)<br>Worker")]
            style BCI_Container fill:#cce5ff,stroke:#007bff,stroke-width:2px
            
            PyTorch[PyTorch / Braindecode]
            GPU[GPU Access (RTX 3090)]
        end
        
        Agent_Env --"1. 指令 (eeg_experiment.py)"--> BCI_Container
        BCI_Container --"2. ログ・結果"--> Agent_Env
        BCI_Container --- PyTorch
        BCI_Container --- GPU
    end