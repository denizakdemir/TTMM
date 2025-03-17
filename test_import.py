print("Testing imports:")
try:
    from tabular_transformer.models.task_heads import MultiTaskHead
    print("✅ Successfully imported MultiTaskHead")
    print(f"MultiTaskHead class: {MultiTaskHead}")
except ImportError as e:
    print(f"❌ Failed to import MultiTaskHead: {e}")
    
    # Try importing directly from the module
    try:
        from tabular_transformer.models.task_heads.multi_task import MultiTaskHead
        print("✅ Successfully imported from direct module path")
        print(f"MultiTaskHead class: {MultiTaskHead}")
    except ImportError as e:
        print(f"❌ Failed to import from direct module path: {e}")
