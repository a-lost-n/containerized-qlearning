image:
	docker build -t freewer/worker:latest worker/.
	docker build -t freewer/master:latest master/.

build:
	kubectl apply -f roles/configMap.yaml
	kubectl apply -f roles/role.yaml
	kubectl apply -f roles/roleAccount.yaml
	kubectl apply -f roles/roleBinding.yaml
	kubectl apply -f worker/worker.yaml
	kubectl apply -f worker/worker-service.yaml
	kubectl apply -f master/master.yaml
	kubectl apply -f master/master-service.yaml
	kubectl apply -f grafana/grafanaConfig.yaml
	kubectl apply -f grafana/grafanaDeployment.yaml
	kubectl apply -f grafana/grafanaService.yaml

monitor:
	kubectl port-forward -n monitoring $(id) 3000 &

pod-info:
	kubectl get pods

clean:
	kubectl delete --all deployment

get-model:
	kubectl cp $(id):/app/model/model.npz /home/freewer/cloud/containerized-qlearning/model/model.npz