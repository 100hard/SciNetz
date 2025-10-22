import { redirect } from "next/navigation";

const DatasetsPage = () => {
  redirect("/papers");
  return null;
};

export default DatasetsPage;
