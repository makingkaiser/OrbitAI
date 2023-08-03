import React, { useState } from "react";
import axios from "axios";
import "./FileUpload.css";
import { RiFileUploadLine } from "react-icons/ri";
import { ReactComponent as ReactLogo } from "./orbit.svg";

axios.defaults.baseURL = 'https://makingorbital.pythonanywhere.com';
export default function FileUpload({ onNamespaceUpdate }) {
  const [file, setFile] = useState(null);
  const [namespace, setNamespace] = useState("");
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [label, setLabel] = useState("Upload a file...");
  const [isLoading, setIsLoading] = useState(false); // Adds a state variable for the loading indicator

  function handleFile(event) {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setErrorMessage(null);
      setSuccessMessage(null);
      setLabel(selectedFile.name);
    } else {
      setFile(null);
      setErrorMessage(null);
      setSuccessMessage(null);
      setLabel("Upload a file...");
    }
  }

  function handleCancel() {
    setFile(null);
    setErrorMessage(null);
    setSuccessMessage(null);
    setLabel("Upload a file...");
  }

  function handleNamespace(event) {
    setNamespace(event.target.value);
    setErrorMessage(null);
  }

  const submitHandler = async (event) => {
    event.preventDefault();
    if (!file && !namespace) {
      const message = "Please select a file or enter a namespace";
      setErrorMessage(message);
      return;
    }

    if (file && namespace) {
      const message =
        "Please select either a file or enter a namespace, not both";
      setErrorMessage(message);
      return;
    }

    if (file) {
      // Check if the selected file is a PDF
      if (file.type !== "application/pdf") {
        const message = "Only PDF files are allowed";
        setErrorMessage(message);
        return;
      }

      // Check if the file size is less than or equal to 50MB
      if (file.size > 50 * 1024 * 1024) {
        const message = "File size should be less than or equal to 50MB";
        setErrorMessage(message);
        return;
      }

      const data = new FormData();
      data.append("file", file);
      console.log(data);

      const namespaceWithGaps = file.name.split(".").slice(0, -1).join("."); // Extract filename without extension


      const namespace = namespaceWithGaps.replace(/\s+/g, "_"); // Replace spaces with underscores

      setIsLoading(true); // Set the loading indicator to true

      try {
        const response = await axios.post(`/upload/${namespace}`, data, {
          headers: {
            "Content-Type": "multipart/form-data", // Set the correct Content-Type header
          },
        });
        console.log("Upload Success!");
        console.log(response.data);
        onNamespaceUpdate(response.data.namespace || namespace); // Pass the namespace value to the onNamespaceUpdate function
        setSuccessMessage(
          "File uploaded successfully! You can now chat with your files."
        ); // Set the success message
        setErrorMessage(null); // Clear any previous error message
      } catch (error) {
        setErrorMessage("An error occurred while uploading the file", error);
        setSuccessMessage(null); // Clear any previous success message
      }

      setIsLoading(false); // Set the loading indicator to false
    } else if (namespace) {
      try {
        const response = await axios.post(`/namespacevalidity/${namespace}`);
        console.log(response.data);
        if (response.data === 1) {
          onNamespaceUpdate(namespace); // Pass the namespace value to the onNamespaceUpdate function
          setSuccessMessage("Namespace submitted successfully!"); // Set the success message
          setErrorMessage(null); // Clear any previous error message
        } else {
          setErrorMessage("Please enter a valid namespace");
          setSuccessMessage(null);
        }
      } catch (error) {
        setErrorMessage(
          "An error occurred while checking the namespace validity",
          error
        );
        setSuccessMessage(null); // Clear any previous success message
      }
    }
  };

  return (
    <div className="landing-page">
      <div className="logo">
        <ReactLogo />
      </div>
      <h1 className="heading">Welcome to OrbitAI! </h1>
      <div className="intro">
        <div id="typewriter"></div>
        <p className="intro-paragraph">
          OrbitAI is your second brain in the cloud. Designed for understanding
          unstructured information.
          <br />
          Ask questions, upload additional documents for OrbitAI to learn,{" "}
          <br /> and share your customized chatbot through namespaces.{" "}
        </p>{" "}
        <div className="intro-paragraph-2">
  <h2>How do I use OrbitAI?</h2>
  <p>
    Simply upload your
    document (currently limited to PDFs) and start chatting with it. OrbitAI
    will answer your questions based on its knowledge of the document's
    contents.
  </p>
  <p>
    Once you've uploaded your document, it will remain in the cloud and can be
    accessed through a generated namespace. This means that you can chat with
    your document from anywhere, on any device. If you close the website but
    want to chat with the document again, you don't have to upload the document
    again; simply enter the namespace.
  </p>
  <p style={{ fontSize: "0.7rem" }}>
          As this is a free instance developed for the CP2106 module, the backend
          is based on a free tier and may take 10-15 seconds to respond.
          Additionally, the API is self-sponsored, so we ask you not to spam our
          bot :( however feel free to try as much as you wish to!
        </p>
</div>

      </div>
      <div className="upload">
        <form action="#" onSubmit={submitHandler}>
          <h2>
            Upload your PDF file (Max 50MB) or enter an existing namespace!
          </h2>
          <label htmlFor="file-upload" className="custom-file-upload">
            <RiFileUploadLine className="upload-icon" />
            {label}
          </label>
          <input
            id="file-upload"
            type="file"
            accept="application/pdf"
            onChange={handleFile}
          />
          {file && (
            <div className="file-actions">
              <button type="button" onClick={handleCancel}>
                Cancel
              </button>
            </div>
          )}
          <input
            type="text"
            placeholder="Enter an existing namespace..."
            value={namespace}
            onChange={handleNamespace}
          />
          <button style={{ fontWeight: "bold" }} disabled={isLoading}>
            Start!
          </button>{" "}
          {/* Disable the button while the file is uploading */}
          {isLoading && <p className="loading">Uploading file...</p>}{" "}
          {/* Show the loading indicator while the file is uploading */}
          {errorMessage && <p className="error">{errorMessage}</p>}
          {successMessage && <p className="success">{successMessage}</p>}
        </form>
      </div>
    </div>
  );
}
