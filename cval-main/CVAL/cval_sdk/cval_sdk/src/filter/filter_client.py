from abc_types import FilterServiceProto


class FilterService(FilterServiceProto):

    def filtering(self, data):
        classification_data = self.filter(data)
        return classification_data

    def send_data_to_annotation_client(self, data, task_config):
        self.annotation_service.send_data_to_service(data, task_config)

