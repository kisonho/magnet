from monai.data.dataloader import DataLoader as _Loader

class DataLoader(_Loader):
    def __contains__(self, value) -> bool:
        for element in self:
            if value == element:
                return True
        return False