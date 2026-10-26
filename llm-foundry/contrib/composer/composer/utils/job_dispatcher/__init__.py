from composer.utils.job_dispatcher.base import JDBase
from composer.utils.job_dispatcher.jd_data import JDData
from composer.utils.job_dispatcher.send_model import JDSendModel


class JD(JDData, JDSendModel):
    pass


__all__ = [
    'JD',
    'JDBase',
    'JDData',
    'JDSendModel',
]
